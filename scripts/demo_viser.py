"""
Visualize a finished PromptHMR run with Viser only (no inference).

Requires results.pkl in the results directory. RGB frames come from (in order):
  1) merged_frames.npz (key "images") in the same folder, or --images-npz
  2) --images-folder (directory of .jpg/.png, same as pipeline image folder)
  3) --video (same clip / settings as the original run: max_height=896, max_fps=60)

Example:
  python scripts/demo_viser.py --results-dir results/forest_riding
  python scripts/demo_viser.py --results-dir results/my_run --video path/to/source.mp4

Joint overlay export / OpenCV step preview are controlled by module constants below
(``_JOINTS_*``), not CLI flags.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import cv2
import joblib
import numpy as np
import torch
import tyro

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from data_config import SMPLX_PATH
from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from prompt_hmr.vis.viser import viser_vis_world4d
from prompt_hmr.vis.traj import get_floor_mesh
from pipeline import Pipeline
from pipeline.utils import load_video_frames


# SMPL-X body joint indices (first 24) + edges for skeleton drawing (parent chain / limbs)
_SMPLX_BODY_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (9, 12),
    (9, 13),
    (9, 14),
    (12, 15),
    (13, 16),
    (16, 18),
    (18, 20),
    (14, 17),
    (17, 19),
    (19, 21),
    (20, 22),
    (21, 23),
]


def _camera_intrinsics_from_results(results: dict) -> np.ndarray:
    cw = results['camera_world']
    fx = float(np.asarray(cw['img_focal']).reshape(-1)[0])
    fy = fx
    cx, cy = np.asarray(cw['img_center'], dtype=np.float64).reshape(2)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


def _project_world_points(
    P_w: np.ndarray,
    T_w2c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """World (N,3) -> image pixels (N,2) with OpenCV-style K applied outside."""
    P_w = np.asarray(P_w, dtype=np.float64).reshape(-1, 3)
    n = P_w.shape[0]
    P_h = np.concatenate([P_w, np.ones((n, 1), dtype=np.float64)], axis=1)
    Pc_h = np.einsum('ij,nj->ni', T_w2c, P_h)
    w = Pc_h[:, 3:4]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    Pc = Pc_h[:, :3] / w
    z = Pc[:, 2]
    valid = z > 1e-4
    return Pc, valid


def _pixels_from_cam_points(Pc: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = Pc[:, 2:3]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * (Pc[:, 0:1] / z) + K[0, 2]
    v = K[1, 1] * (Pc[:, 1:2] / z) + K[1, 2]
    return np.concatenate([u, v], axis=1)


def render_joints_overlay(
    joints_world: np.ndarray,
    T_w2c_4x4: np.ndarray,
    K: np.ndarray,
    image_shape_hw: tuple[int, int],
    background_rgb: np.ndarray | None = None,
    edges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Project world joints with ``T_w2c`` + ``K``, draw skeleton on RGB image.

    If ``background_rgb`` is None, draws on a black canvas (H,W,3) uint8 RGB.
    Otherwise ``background_rgb`` must match ``image_shape_hw`` (same as the loaded video frame).
    """
    if edges is None:
        edges = _SMPLX_BODY_EDGES
    h, w = int(image_shape_hw[0]), int(image_shape_hw[1])
    if background_rgb is None:
        canvas_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        bg = np.asarray(background_rgb, dtype=np.uint8)
        if bg.shape[0] != h or bg.shape[1] != w:
            raise ValueError(f'background_rgb shape {bg.shape[:2]} != {(h, w)}')
        canvas_bgr = cv2.cvtColor(bg.copy(), cv2.COLOR_RGB2BGR)

    n_p, n_j, _ = joints_world.shape
    # BGR for OpenCV
    colors_bgr = [
        (0, 255, 0),
        (0, 165, 255),
        (255, 200, 0),
        (255, 0, 255),
        (0, 200, 200),
    ]
    for pid in range(n_p):
        col = colors_bgr[pid % len(colors_bgr)]
        j3 = joints_world[pid, :, :]
        Pc, valid = _project_world_points(j3, T_w2c_4x4)
        uv = _pixels_from_cam_points(Pc, K)
        for a, b in edges:
            if a >= n_j or b >= n_j:
                continue
            if not (valid[a] and valid[b]):
                continue
            pa = (int(round(uv[a, 0])), int(round(uv[a, 1])))
            pb = (int(round(uv[b, 0])), int(round(uv[b, 1])))
            if 0 <= pa[0] < w and 0 <= pa[1] < h and 0 <= pb[0] < w and 0 <= pb[1] < h:
                cv2.line(canvas_bgr, pa, pb, col, 2, lineType=cv2.LINE_AA)
        for j in range(n_j):
            if not valid[j]:
                continue
            pt = (int(round(uv[j, 0])), int(round(uv[j, 1])))
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(canvas_bgr, pt, 4, col, -1, lineType=cv2.LINE_AA)
    return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)


def _load_rgb_frames(
    results_dir: str,
    results: dict,
    images_npz: Optional[str],
    images_folder: Optional[str],
    video: Optional[str],
) -> np.ndarray:
    n_expect = len(results['camera']['pred_cam_R'])

    npz_path = images_npz or os.path.join(results_dir, 'merged_frames.npz')
    if os.path.isfile(npz_path):
        images = np.load(npz_path)['images']
        if len(images) < n_expect:
            raise ValueError(
                f'{npz_path} has {len(images)} frames but results.pkl expects {n_expect}.'
            )
        if len(images) > n_expect:
            images = images[:n_expect]
        return images

    if images_folder is not None:
        if not os.path.isdir(images_folder):
            raise FileNotFoundError(images_folder)
        imgs, _, _ = load_video_frames(
            images_folder,
            output_folder=results_dir,
            max_height=896,
            max_fps=60,
        )
    elif video is not None:
        if not os.path.isfile(video):
            raise FileNotFoundError(video)
        imgs, _, _ = load_video_frames(
            video,
            output_folder=results_dir,
            max_height=896,
            max_fps=60,
        )
    else:
        raise FileNotFoundError(
            f'No merged_frames.npz in {results_dir}. '
            'Pass --video (source clip) or --images-folder (extracted frames), '
            'or --images-npz pointing to an .npz with array "images".'
        )

    if len(imgs) < n_expect:
        raise ValueError(
            f'Loaded {len(imgs)} RGB frames but results.pkl has {n_expect} camera frames.'
        )
    if len(imgs) > n_expect:
        imgs = imgs[:n_expect]
    return imgs


def main(
    results_dir: str = 'results/forest_riding',
    video: Optional[str] = None,
    images_folder: Optional[str] = None,
    images_npz: Optional[str] = None,
    fps: Optional[float] = None,
    static_camera: bool = False,
    viser_subsample: int = 1,
    viser_total: Optional[int] = None,
):
    results_dir = os.path.expanduser(results_dir)
    pkl_path = os.path.join(results_dir, 'results.pkl')
    if not os.path.isfile(pkl_path):
        sys.stderr.write(
            f'Missing {pkl_path}.\n'
            'Viser needs the pickled pipeline output. '
            'If you only have world4d.mcs, open that in Meshcapade / Blender instead.\n'
        )
        raise SystemExit(1)

    results = joblib.load(pkl_path)
    images = _load_rgb_frames(results_dir, results,
                              images_npz, images_folder, video)

    if fps is None:
        fps = float(results.get('merged_fps', 30.0))

    smplx = SMPLX_Layer(SMPLX_PATH).cuda()
    vis_pipeline = Pipeline(static_cam=static_camera)
    vis_pipeline.results = results
    vis_pipeline.images = images
    vis_pipeline.fps = fps
    vis_pipeline.cfg.fps = fps
    vis_pipeline.cfg.seq_folder = results_dir

    n_all = len(images)
    cap = viser_total if viser_total is not None else n_all
    cap = min(cap, n_all)
    images_vis = images[:cap][::viser_subsample]
    world4d = vis_pipeline.create_world4d(step=viser_subsample, total=cap)
    world4d = {i: world4d[k] for i, k in enumerate(sorted(world4d.keys()))}

    K = _camera_intrinsics_from_results(results)

    all_verts = []
    frame_keys = sorted(world4d.keys(), key=int)
    for k in frame_keys:
        world3d = world4d[k]
        ki = int(k)
        frame_rgb = images_vis[ki]
        h0, w0 = frame_rgb.shape[:2]

        rotmat = axis_angle_to_matrix(world3d['pose'].reshape(-1, 55, 3))
        smpl_out = smplx(
            global_orient=rotmat[:, :1].cuda(),
            body_pose=rotmat[:, 1:22].cuda(),
            betas=world3d['shape'].cuda(),
            transl=world3d['trans'].cuda(),
        )
        joints = smpl_out.joints.cpu().numpy()
        verts = smpl_out.vertices.cpu().numpy()

        cam4 = np.asarray(world3d['camera'], dtype=np.float64)
        T_w2c = np.eye(4, dtype=np.float64)
        T_w2c[:3, :4] = cam4[:3, :4]


        out_r = render_joints_overlay(
            joints, T_w2c, K, (h0, w0), background_rgb=frame_rgb
        )
        cv2.imshow('frame', cv2.cvtColor(out_r, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

        world3d['vertices'] = verts
        all_verts.append(torch.tensor(verts, dtype=torch.bfloat16))

    floor_arg = None
    if len(all_verts) > 0:
        merged_v = torch.cat(all_verts)
        gv, gf, _ = get_floor_mesh(merged_v, scale=2)
        floor_arg = [gv, gf]

    server, gui = viser_vis_world4d(
        images_vis,
        world4d,
        smplx.faces,
        floor=floor_arg,
        init_fps=max(1.0, fps / viser_subsample),
    )

    url = f'https://localhost:{server.get_port()}'
    print(f'Open: {url}')
    gui_playing, gui_timestep, gui_framerate, num_frames = gui
    while True:
        # 不按 Playing 自动推进时间轴（只用 Viser 里手动拖 / 点播放）
        # if gui_playing.value:
        #     gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)


if __name__ == '__main__':
    tyro.cli(main)
