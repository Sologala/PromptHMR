"""
Visualize a finished PromptHMR run with Viser only (no inference).

Requires results.pkl in the results directory. RGB frames come from (in order):
  1) merged_frames.npz (key "images") in the same folder, or --images-npz
  2) --images-folder (directory of .jpg/.png, same as pipeline image folder)
  3) --video (same clip / settings as the original run: max_height=896, max_fps=60)

Example:
  python scripts/demo_viser.py --results-dir results/forest_riding
  python scripts/demo_viser.py --results-dir results/my_run --video path/to/source.mp4
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

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
    images = _load_rgb_frames(results_dir, results, images_npz, images_folder, video)

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

    all_verts = []
    for k in world4d:
        world3d = world4d[k]
        if len(world3d['track_id']) == 0:
            continue
        rotmat = axis_angle_to_matrix(world3d['pose'].reshape(-1, 55, 3))
        verts = smplx(
            global_orient=rotmat[:, :1].cuda(),
            body_pose=rotmat[:, 1:22].cuda(),
            betas=world3d['shape'].cuda(),
            transl=world3d['trans'].cuda(),
        ).vertices.cpu().numpy()
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
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)


if __name__ == '__main__':
    tyro.cli(main)
