from collections import defaultdict
import os
import shutil
import cv2
import joblib
import numpy as np
import torch
from omegaconf import OmegaConf

from smplx import SMPLX
from pipeline.detector import segment
from pipeline.detector.segment import load_masks_from_disk
from pipeline.detector.vitpose_estimator import load_vit_model, estimate_kp2ds_from_bbox_vitpose
from pipeline.kp_utils import convert_kps
from pipeline.utils import prepare_inputs, load_video_frames, interpolate_bboxes
from pipeline.tools import detect_track, detect_segment_track_sam, est_camera, est_calib
from pipeline.phmr_vid import PromptHMR_Video
from pipeline.camera import run_metric_slam, calibrate_intrinsics, run_slam
from pipeline.spec import run_cam_calib
from pipeline.world import world_hps_estimation
from pipeline.postprocessing import post_optimization
from pipeline.mcs_export_cam import export_scene_with_camera
from smplcodec import SMPLCodec
from pipeline.streaming_dataset import StreamingVideoDataset
from pipeline.chunked_processing import ChunkedVideoProcessor, save_intermediate_results

from ultralytics import YOLO
from boxmot import BotSort
from pathlib import Path
import pickle
from tqdm import tqdm

class Pipeline:
    def __init__(self, static_cam=False, shift_fps=10, use_streaming=False, 
                 use_chunked_processing=False, chunk_size=300, chunk_overlap=30):
        self.images = None
        self.images_list = []
        self.mask_list = []
        self.cfg = OmegaConf.load("pipeline/config.yaml")
        self.cfg.static_cam = static_cam
        self.shrink_fps = shift_fps
        self.use_streaming = use_streaming
        self.use_chunked_processing = use_chunked_processing
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunked_processor = ChunkedVideoProcessor(chunk_size, chunk_overlap)

        checkpoint_dir = 'data/pretrain'
        self.data_dict = {
            'droid': os.path.join(checkpoint_dir, 'droid.pth'), 
            'sam': os.path.join(checkpoint_dir, "sam_vit_h_4b8939.pth"), 
            'sam2': os.path.join(checkpoint_dir, "sam2_ckpts"), 
            'yolo': os.path.join(checkpoint_dir, 'yolo11x.pt'), 
            'vitpose': os.path.join(checkpoint_dir, 'vitpose-h-coco_25.pth'), 
        }

        self.smplx = SMPLX(
            f'data/body_models/smplx/SMPLX_NEUTRAL.npz', 
            use_pca=False, 
            flat_hand_mean=True, 
            num_betas=10
        )
        self.yolo = YOLO("data/yolo11x.pt")
        

    def get_frames_as_numpy(self, indices=None):
        """
        Get frames as numpy array. If using streaming, loads on-demand.
        If indices=None, returns all frames (memory intensive!).
        """
        if isinstance(self.images, np.ndarray):
            # Already in memory
            if indices is None:
                return self.images
            else:
                return self.images[indices]
        else:
            # File paths - load from disk
            import cv2
            if indices is None:
                indices = list(range(len(self.images)))
            
            frames = []
            for idx in indices:
                frame = cv2.imread(self.images[idx])
                if frame is None:
                    raise RuntimeError(f"Failed to load image: {self.images[idx]}")
                frames.append(frame[..., ::-1])  # BGR to RGB
            return np.array(frames)
    
    def get_frame_count(self):
        """Get total number of frames."""
        if isinstance(self.images, np.ndarray):
            return len(self.images)
        else:
            return len(self.images) if self.images else 0
        # If using streaming, save frames to disk instead of loading into memory
        if self.use_streaming:
            images, seq_folder, img_folder, fps = prepare_inputs(
                input_video,
                output_folder=output_folder,
                max_height=896,
                max_fps=60,
            )
            # Return image folder path instead of numpy array for streaming
            self.fps = fps
            return images, seq_folder
        
        if read_frames == True:
            images, seq_folder, fps = load_video_frames(
                input_video,
                output_folder=output_folder,
                max_height=896,
                max_fps=60,
            )
        else:
            # this currently will cause issue with sam2
            images, seq_folder, img_folder, fps = prepare_inputs(
                input_video,
                output_folder=output_folder,
                max_height=896,
                max_fps=60,
            )
        self.fps = fps
        return images, seq_folder

    def run_detect_track(self, ):
        print("run segmenting")
        mask_dir = os.path.join(self.output_dir, 'masks')
        masks = segment.segment_subjects(
            self.images_list, mask_dir=mask_dir)
        print("segmenting done")

        self.results['masks'] = masks
        self.results['masks_are_paths'] = isinstance(
            masks, list) and len(masks) > 0 and isinstance(masks[0], str)
        self.results['mask_dir'] = os.path.join(self.output_dir, 'masks') if isinstance(
            masks, list) and len(masks) > 0 and isinstance(masks[0], str) else None


        

    def estimate_2d_keypoints(self,):
        model = load_vit_model(
            model_path='data/pretrain/vitpose-h-coco_25.pth')
        
        for k, v in self.results['people'].items():
            kpts_2d = estimate_kp2ds_from_bbox_vitpose(
                model, self.images_list, v['bboxes'], k, v['frames'])
            kpts_2d = convert_kps(kpts_2d, 'vitpose25', 'openpose')
            self.results['people'][k]['keypoints_2d'] = kpts_2d
            coco_kp2d = convert_kps(kpts_2d, 'ophandface', 'cocoophf')
            self.results['people'][k]['vitpose'] = coco_kp2d

        self.results['has_2d_kpts'] = True
        del model
        return

    def visualize_results(self,):
        video_path = os.path.join(self.output_dir, 'vis.mp4')
        print(f'Start writing visualization to {video_path} ...')
        
        colors = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
            (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128),
        ]
        
        first_frame = cv2.imread(self.images_list[0])
        h, w = first_frame.shape[:2]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        people = self.results['people']
        has_keypoints = self.results.get('has_2d_kpts', False)
        
        for frame_idx in tqdm(range(len(self.images_list)), desc='Rendering visualization'):
            img = cv2.imread(self.images_list[frame_idx])
            
            for pid, person in people.items():
                color = colors[int(pid) % len(colors)]
                frames = person['frames']
                in_frame = np.where(frames == frame_idx)[0]
                
                if len(in_frame) == 0:
                    continue
                
                idx = in_frame[0]
                bbox = person['bboxes'][idx]
                x1, y1, x2, y2 = map(int, bbox[:4])
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{int(pid)}"
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if has_keypoints and 'keypoints_2d' in person:
                    kpts = person['keypoints_2d'][idx]
                    for kp in kpts:
                        if kp[2] > 0.3:
                            cv2.circle(img, (int(kp[0]), int(kp[1])), 3, color, -1)
            
            out.write(img)
        
        out.release()
        print(f"Visualization saved to {video_path}")

    def hps_estimation(self,):
        if self.cfg.tracker == 'sam2':
            mask_prompt = True
        else:
            mask_prompt = False

        # Determine mask cache directory
        mask_cache_dir = None
        if self.use_streaming:
            mask_cache_dir = os.path.join(self.output_dir, 'mask_cache')
        elif self.results.get('masks_are_paths', False):
            # Use the existing mask directory for streaming-like access
            mask_cache_dir = self.results.get('mask_dir')
        
        phmr = PromptHMR_Video(
            use_streaming=self.use_streaming,
            mask_cache_dir=mask_cache_dir
        )
        self.results = phmr.run(
            self.images_list, self.results, mask_prompt,
            use_streaming=self.use_streaming,
            mask_cache_dir=mask_cache_dir
        )
        self.results['contact_joint_ids'] = [7, 10, 8, 11, 20, 21]
        self.results['has_hps_cam'] = True
    
        return


    def camera_motion_estimation(self, static_cam = False):
        ##### Run Masked DROID-SLAM #####
        
        opt_intr = False if self.cfg.use_depth else True
        keyframes = None

        if self.cfg.static_cam or static_cam:
            print("Using static camera assumption")
            static_cam = True
            if self.cfg.calib is None:
                # Only need first frame for calib estimation, avoid loading all frames
                first_frame = cv2.imread(self.images_list[0])[..., ::-1]
                cam_int = est_calib([first_frame])
            else:
                cam_int = np.loadtxt(self.cfg.calib)
                opt_intr = False

        else:
            # Need all frames and masks for SLAM
            if isinstance(self.images, list):
                print(f"Loading frames for camera motion estimation...")
                images_data = self.get_frames_as_numpy()
            else:
                images_data = self.images
            
            masks = self.results['masks']
            if self.results.get('masks_are_paths', False):
                print(f"Loading masks from disk...")
                masks = load_masks_from_disk(masks)
            masks = torch.from_numpy(masks)
            
            assert masks.shape[0] == len(images_data), f"Masks and images should be same length {masks.shape[0]} != {len(images_data)}"
            
            if self.cfg.calib is None:
                if self.cfg.focal is None and opt_intr == False:
                    try:
                        if self.cfg.calibMethod == 'ba':
                            _, _, cam_int, keyframes = run_slam(
                                images_data, masks=masks, opt_intr=True, 
                                stride=self.cfg.calib_stride,
                            )
                        elif self.cfg.calibMethod == 'iterative':    
                            cam_int = calibrate_intrinsics(self.cfg.img_folder, masks)
                    except ValueError as e:
                        static_cam = True
                        print(e)
                        print("Warning: probably there is not much camera motion in the video!!")
                        cam_int = est_calib(images_data)

                elif self.cfg.focal is not None:
                    cam_int = est_calib(images_data)
                    cam_int[0] = self.cfg.focal
                    cam_int[1] = self.cfg.focal
                    opt_intr = False
                else:
                    cam_int = est_calib(images_data)
            else:
                cam_int = np.loadtxt(self.cfg.calib)
                opt_intr = False
        
        if static_cam:
            total_frames = self.get_frame_count()
            cam_R = torch.eye(3)[None].repeat_interleave(total_frames, 0)
            cam_T = torch.zeros((total_frames, 3))
            print("Warning: probably there is not much camera motion in the video!!")
            print("Setting camera motion to zero")
        else:
            try:
                cam_R, cam_T, cam_int = run_metric_slam(
                    images_data, 
                    masks=masks, 
                    calib=cam_int, 
                    monodepth_method=self.cfg.depth_method, 
                    use_depth_inp=self.cfg.use_depth,
                    stride=self.cfg.stride,
                    opt_intr=opt_intr,
                    save_depth=self.cfg.save_depth,
                    keyframes=keyframes,
                )
            except ValueError as e:
                if str(e).startswith("not enough values to unpack"):
                    cam_R = torch.eye(3)[None].repeat_interleave(len(masks), 0)
                    cam_T = torch.zeros((len(masks), 3))
                    print("Warning: probably there is not much camera motion in the video!!")
                    print("Setting camera motion to zero")
                else:
                    raise e
                    
        print("Camera intrinsics:", cam_int)
        camera = {
            'pred_cam_R': cam_R.numpy(), 
            'pred_cam_T': cam_T.numpy(), 
            'img_focal': cam_int[0], 
            'img_center': cam_int[2:]
        }
        print("cam focal length: ", cam_int[0])
        self.results['camera'] = camera
        self.results['has_slam'] = True
        return


    def world_hps_estimation(self, ):
        self.results = world_hps_estimation(self.cfg, self.results, self.smplx)
        self.results['has_hps_world'] = True
        return
    

    def post_optimization(self):
        self.results = post_optimization(
            self.cfg, self.results, self.images_list, 
            self.smplx, opt_contact=True,
        )
        self.results['has_post_opt'] = True


    def get_K(self, ):
        camera = self.results['camera']
        K = np.eye(3)
        K[0,0] = camera['img_focal']
        K[1,1] = camera['img_focal']
        K[:2,-1] = camera['img_center']
        K = torch.tensor(K, dtype=torch.float)
        return K
    

    def create_world4d(self, results=None, total=None, step=1):
        if results is None:
            results = self.results
        if total is None:
            total = len(results['camera']['pred_cam_R'])
        else:
            total = min(total, len(results['camera']['pred_cam_R']))
            
        world4d = {}
        for i in range(0, total, step):
            pose = []
            shape = []
            transl = []
            track_id = []

            # People
            for pid in results['people']:
                people = results['people'][pid]
                frames = people['frames']
                in_frame = np.where(frames == i)[0]

                if len(in_frame) == 1:
                    smplx_w = people['smplx_world']
                    pose.append(smplx_w['pose'][in_frame])
                    shape.append(smplx_w['shape'][in_frame])
                    transl.append(smplx_w['trans'][in_frame])
                    track_id.append(people['track_id'])
            
            # Camera
            camera_w = results['camera_world']
            Rwc = camera_w['Rwc'][i]
            Twc = camera_w['Twc'][i]
            camera = np.eye(4)
            camera[:3,:3] = Rwc
            camera[:3, 3] = Twc

            if len(track_id) > 0:
                world4d[i] = {'pose': torch.tensor(np.concatenate(pose)).float().reshape(len(track_id),-1,3),
                            'shape': torch.tensor(np.concatenate(shape)).float(),
                            'trans': torch.tensor(np.concatenate(transl)).float(),
                            'track_id': torch.tensor(np.array(track_id)) - 1,
                            'camera': camera}
            else:
                world4d[i] = {'track_id': np.array([]),
                            'camera': camera}

        return world4d

    def convert_video_to_frames(self, input_video, output_folder):
        self.frame_img_path = output_folder +  '/' + "frames"
        os.makedirs(self.frame_img_path, exist_ok=True)

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise FileNotFoundError(
                f"Error: Could not open video file: {input_video}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.images_list = []
        for i in tqdm(range(frame_count), desc="Converting video to frames"):
            frame_img_path = os.path.join(self.frame_img_path, f"{i:08d}.jpg")
            frame_img_path = os.path.abspath(frame_img_path)

            if os.path.isfile(frame_img_path):
                # skip existing frame
                self.images_list.append(frame_img_path)
                continue

            ret, frame = cap.read()
            cv2.imwrite(frame_img_path, frame)
            self.images_list.append(frame_img_path)
        print(
            f"converted {len(self.images_list)} frames to output folder: {self.frame_img_path}")

    def __call__(self, input_video, output_folder, static_cam=False,
                 save_only_essential=False):
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise FileNotFoundError(
                f"Error: Could not open video file: {input_video}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        all_clip_info_path = os.path.join(output_folder, "clip_infos.pkl")
        clips = [] 
        if os.path.exists(all_clip_info_path):
            clips = self.load_and_clip_infos(all_clip_info_path)
        else:

            all_trks = self.process_all_track_by_boxmot(cap)
            len_trks = len(all_trks)
            if len_trks == 0:
                print("No track found, skip this video")
                return
            clips = self.clip_video_by_detect_and_tracking(
                cap, output_folder, all_trks, all_clip_info_path)

        for clip in clips:
            clip_output= clip['output_dir']
            self.process_on_clip(clip, clip_output, static_cam, save_only_essential)
        
        pass
    
    def load_and_clip_infos(self, clip_info_path):
        with open(clip_info_path, "rb") as f:
            clips = pickle.load(f)

        print(f"loading clip infos... {len(clips)} clips")
        return clips

    def process_all_track_by_boxmot(self, cap: cv2.VideoCapture):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        mot = BotSort(
            reid_weights=Path("osnet_x0_25_msmt17.pt"),
            device="0",
            half=False,
        )

        trk_results = defaultdict(lambda: defaultdict(list))
        fid = 0
        ntotal_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fid in tqdm(range(ntotal_frames), desc="Tracking"):
            ok, frame = cap.read()
            if not ok:
                break

            results = self.yolo(frame, verbose=False, classes=0,
                                device="0", conf=0.25)[0]

            detections = np.empty((0, 6), dtype=np.float32)
            if results.boxes is not None and len(results.boxes) > 0:
                detections = results.boxes.data.cpu().numpy()

            tracks = mot.update(detections, frame)

            for track in tracks:
                x1, y1, x2, y2, id, conf, cls, det_ind = track
                area = (x2 - x1) * (y2 - y1)
                if area < 100:
                    continue

                box_xyxy = [x1, y1, x2, y2]
                trk_results[id]['bboxes'].append(np.array(box_xyxy))
                trk_results[id]['frames'].append(fid)
                
        for tid in trk_results:
            trk_results[tid]['track_id'] = tid
            trk_results[tid]['detected'] = len(trk_results[tid]['frames'])

        return trk_results



    def clip_video_by_detect_and_tracking(self, cap, output_folder, all_trks, clip_info_path):
        min_frames = 10
        min_continues_frames = 30

        trkd_frames = defaultdict(list)
        for id in all_trks:
            for fid in all_trks[id]['frames']:
                trkd_frames[fid].append(id)

        # find continuous frames with tracks used
        fids = list(trkd_frames.keys())
        fids.sort()
        # 筛选连续帧段
        clips = []
        

        if fids:
            current_segment = [fids[0]]
            for i in range(1, len(fids)):
                # 检查是否连续（帧ID相差1）
                if fids[i] == fids[i-1] + 1:
                    current_segment.append(fids[i])
                else:
                    # 连续中断，保存当前段并开始新段
                    if len(current_segment) >= min_continues_frames:  # 确保段长度足够
                        clips.append(current_segment)
                    current_segment = [fids[i]]

            # 保存最后一个段
            if len(current_segment) >= min_continues_frames:
                clips.append(current_segment)

        # 处理每个连续帧段
        print("total clips:", len(clips))
        all_clips = []
        for clip_id, clip in enumerate(clips):
            clip_output_path = os.path.join(
                output_folder, f"clip_{clip_id:02d}")
            os.makedirs(clip_output_path, exist_ok=True)

            clip_frames_path = os.path.join(clip_output_path, "frames")
            os.makedirs(clip_frames_path, exist_ok=True)
            clip_masks_path = os.path.join(clip_output_path, "masks")
            os.makedirs(clip_masks_path, exist_ok=True)
            
            clip_info = {
                "output_dir": clip_output_path,
                "clip_start_frame": clip[0],
                "frames_count": len(clip),
                "frame_ids": clip,
                "images_list": [],
                "track_ids": [],
                "track_info": None
            }

            cap.set(cv2.CAP_PROP_POS_FRAMES, clip[0])

            trk_ids_in_clip = set()
            for fid in clip:
                # 0. 保存图片到文件夹
                frame_img_path = os.path.join(
                    clip_frames_path, f"{fid:08d}.jpg")
                frame_img_path = os.path.abspath(frame_img_path)
                ret, frame = cap.read()
                cv2.imwrite(frame_img_path, frame)

                clip_info["images_list"].append(frame_img_path)

                for trkid in trkd_frames[fid]:
                    trk_ids_in_clip.add(trkid)

            trks_in_clip = {}
            # 删除这些id中不在当前clip中的box 和 frames id
            ids_set = set(clip)
            total_boxes = 0
            for trkid in trk_ids_in_clip:
                trks = all_trks[trkid]
                if trkid not in trks_in_clip:
                    trks_in_clip[trkid] = {'bboxes': [], 'frames': []}
                for k in range(len(trks['frames'])):
                    if trks['frames'][k] in ids_set:
                        trks_in_clip[trkid]['bboxes'].append(trks['bboxes'][k])
                        trks_in_clip[trkid]['frames'].append(trks['frames'][k])
                        total_boxes += 1
            
            for tid in trks_in_clip:
                trks_in_clip[tid]['track_id'] = tid
                trks_in_clip[tid]['detected'] = len(trks_in_clip[tid]['frames'])
            
            clip_info["track_ids"] = list(trk_ids_in_clip)
            clip_info["track_info"] = trks_in_clip
            
            print(
                f"Clip {clip_id+1}:\n\t frames {clip[0]}-{clip[-1]} ({len(clip)} frames), trkids: {len(trk_ids_in_clip)}, total boxes: {total_boxes}")
            all_clips.append(clip_info)

        with open(clip_info_path, "wb") as f:
            print(f"dump clip info to {clip_info_path}")
            pickle.dump(all_clips, f)
        return all_clips

    def process_on_clip(self, clip_info, output_folder, static_cam=False,
                           save_only_essential=False):
        import time
        start_time = time.time()

        print(f"Processing clip {clip_info['output_dir']}...")


        def cvt_to_numpy(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    cvt_to_numpy(v)
                elif isinstance(v, torch.Tensor):
                    d[k] = v.detach().cpu().numpy()
        cvt_to_numpy(clip_info)

        if 'clip_start_frame' not in clip_info:
            clip_info["clip_start_frame"] = int(clip_info["frame_ids"][0])

        self.images_list = clip_info["images_list"]
        self.images = self.images_list
        self.output_dir = clip_info["output_dir"]

        self.results = {
            'camera': {},
            'people': {},
            'timings': {},
            'masks': None,
            'has_tracks': False,
            'has_hps_cam': False,
            'has_hps_world': False,
            'has_slam': False,
            'has_hands': False,
            'has_2d_kpts': False,
            'has_post_opt': False,
        }

        # naive camera
        if not self.results['has_slam']:
            # Load first frame for camera estimation
            first_frame = cv2.imread(
                self.images_list[0])[..., ::-1]  # BGR to RGB
            self.results['camera'] = est_camera(first_frame)

        # spec camera
        if not self.results['has_slam']:
            stride = len(self.images_list)//30
            if stride == 0:
                stride = 1
            spec_calib = run_cam_calib(self.images_list, out_folder=self.output_dir+'/spec_calib',
                                       save_res=True, stride=stride, method='spec',
                                       first_frame_idx=0)
            self.results['spec_calib'] = spec_calib
            print("the focallenth of camera calib is ",
                  spec_calib["median_focal_length"])

        # detect_segment_track
        if not self.results['has_tracks']:
            print("Running segment, and track pipeline...")
            self.run_detect_track()

        self.results['people'] = clip_info["track_info"]
        self.results['has_tracks'] = True

        # Remap frame indices from global video indices to local clip indices
        first_frame_idx = clip_info["clip_start_frame"]
        for tid, track in self.results['people'].items():
            if 'frames' in track:
                track['frames'] = np.array(track['frames']) - first_frame_idx

        # Ensure all track data is numpy (not list)
        for tid, track in self.results['people'].items():
            for key in ['bboxes', 'frames', 'keypoints_2d', 'vitpose']:
                if key in track and isinstance(track[key], list):
                    track[key] = np.array(track[key])

        # slam
        if not self.results['has_slam']:
            print("Running camera motion estimation...")
            self.camera_motion_estimation(static_cam)

        # keypoints detection
        if not self.results['has_2d_kpts']:
            print("Estimating 2D keypoints...")
            self.estimate_2d_keypoints()

        # visualization (after detect + segment + keypoints)
        print("Generating visualization...")
        self.visualize_results()

        # hps
        if not self.results['has_hps_cam']:
            print("Running human mesh estimation...")
            self.hps_estimation()

        # convert hps to world coordinate
        if not self.results['has_hps_world']:
            print("Running world coordinates estimation...")
            self.world_hps_estimation()

        cvt_to_numpy(self.results)

        # ### post optimization
        # if self.cfg.run_post_opt and not self.results['has_post_opt']:
        #     print("Running post optimization...")
        #     self.post_optimization()

        if save_only_essential:
            _ = self.results.pop('masks', None)
            for tid, track in self.results['people'].items():
                _ = track.pop('masks', None)
                _ = track.pop('keypoints_2d', None)
                _ = track.pop('vitpose', None)
                _ = track.pop('prhmr_img_feats', None)
                
        joblib.dump(self.results, f'{self.output_dir}/results.pkl')
        
        NUM_FRAMES = len(self.images)
        MCS_OUTPUT_PATH = f'{self.output_dir}/world4d.mcs'
        smpl_paths = []
        per_body_frame_presence = []
        for k,v in self.results['people'].items():
            out_smpl_f = f'{os.path.abspath(self.output_dir)}/subject-{k}.smpl'
            
            SMPLCodec(
                shape_parameters=v['smplx_world']['shape'].mean(0),
                body_pose=v['smplx_world']['pose'][:, :22*3].reshape(-1,22,3), 
                body_translation=v['smplx_world']['trans'],
                frame_count=v['frames'].shape[0], frame_rate=float(self.cfg.fps)
            ).write(out_smpl_f)
            smpl_paths.append(out_smpl_f)
            per_body_frame_presence.append([int(v['frames'][0]), int(v['frames'][-1])+1])
        
        export_scene_with_camera(
            smpl_buffers=[open(path, 'rb').read() for path in smpl_paths],
            frame_presences=per_body_frame_presence,
            num_frames=NUM_FRAMES,
            output_path=MCS_OUTPUT_PATH,
            rotation_matrices=self.results['camera_world']['Rcw'],
            translations=self.results['camera_world']['Tcw'],
            focal_length=self.results['camera_world']['img_focal'],
            principal_point=self.results['camera_world']['img_center'],
            frame_rate=float(self.cfg.fps),
            smplx_path='data/body_models/smplx/SMPLX_neutral_array_f32_slim.npz',
        )

        print("Usage:")
        print(f'\tYou can drag and drop the "world4d.mcs" file to https://me.meshcapade.com/editor to view the result')
        print(f'\tYou can import the "world4d.glb" file on Blender to view the result')
        
        elapsed = time.time() - start_time
        num_frames = len(self.images_list)
        print("\n" + "="*60)
        print(f"Processing Complete")
        print(f"  Start time : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print(f"  Total frames: {num_frames}")
        print(f"  Elapsed    : {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"  FPS        : {num_frames/elapsed:.2f}")
        print("="*60)
        
        return self.results