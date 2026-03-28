import os
from typing import Union
import cv2
import numpy as np
import torch
import time
import tyro
from pprint import pprint
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')

from pipeline.spec import run_cam_calib
from pipeline.tools import detect_track, detect_segment_track_sam, est_camera, est_calib
from pipeline import Pipeline



class PipelineChunkInferMerge(Pipeline):
    def __init__(self, static_cam=True):
        super().__init__(static_cam)
        self.cap = None
        self.frame_index = 0
        self.chunk_size = 1000

    def load_frames(self, input_video, output_folder, read_frames=True):
        return super().load_frames(input_video, output_folder, read_frames)

    def load_video_to_cap(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.video_info = {
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            "framecount": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }

        pprint(self.video_info)
        seq = os.path.basename(video_path).split('.')[0]
        self.seq_folder = f'results/{seq}'
        os.makedirs(self.seq_folder, exist_ok=True)

    def get_frame(self, frame_index: Union[int, list], numpy =False):
        if isinstance(frame_index, int):
            frame_indices = [frame_index]
        else:
            frame_indices = frame_index

        frames = []
        for idx in frame_indices:
            if (idx < 0 or idx >= self.video_info["framecount"]):
                raise ValueError(f"Frame index {idx} is out of bounds")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame {idx}")
            frames.append(frame)
        if numpy:
            frames = np.array(frames)
        return frames

    def __call__(self, input_video, output_folder, save_only_essential=False, max_frame=None):
        self.load_video_to_cap(input_video)

        def cvt_to_numpy(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    cvt_to_numpy(v)
                elif isinstance(v, torch.Tensor):
                    d[k] = v.detach().cpu().numpy()

        print("output_folder:", output_folder)
        print(self.data_dict)

        self.cfg.seq_folder = self.seq_folder

        # create init results
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

        # 0. 创建相机
        if not self.results['has_slam']:
            self.results['camera'] = est_camera(self.get_frame(0)[0])
            print("estimate camera:", self.results['camera'])

        # 1. 标定相机内参
        if not self.results['has_slam']:
            stride = self.video_info["framecount"] // 30
            max_frame_count_for_calib = 100
            if stride == 0:
                stride = 1

            # 生成 0 - self.framecount的最多100张图片均匀采样。
            frame_indexs = list(
                range(0, self.video_info["framecount"], stride))
            frame_indexs = frame_indexs[:max_frame_count_for_calib]
            calib_images = self.get_frame(frame_indexs, numpy=True)
            # for img in calib_images:
            #     cv2.imshow("frame", img)
            #     cv2.waitKey(30)

            spec_calib = run_cam_calib(calib_images, out_folder=self.seq_folder+'/spec_calib',
                                       save_res=True, stride=1, method='spec',
                                       first_frame_idx=0)
            self.results['spec_calib'] = spec_calib
            print("estimate spec calib:",
                  self.results['spec_calib']["median_focal_length"])

        # 2. 检测，跟踪，分割
        if not self.results['has_tracks']:
            print("Running detect, segment, and track pipeline...")
            self.run_detect_track()





        pass


def main(input_video='data/examples/boxing_short.mp4',
         run_viser=False,
         viser_total=1500,
         viser_subsample=1,
         output_folder=None):
    print("starting demo_video.py")
    if output_folder is None:
        output_folder = 'results/' + \
            os.path.basename(input_video).split('.')[0]
    else:
        output_folder = output_folder

    pipeline = PipelineChunkInferMerge(static_cam=True)
    results = pipeline.__call__(input_video,
                                output_folder,
                                save_only_essential=False)
    pass


if __name__ == '__main__':
    tyro.cli(main)
