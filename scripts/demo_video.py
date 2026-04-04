import os
import sys
import cv2
import numpy as np
import torch
import time
import tyro

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from data_config import SMPLX_PATH
from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from prompt_hmr.vis.viser import viser_vis_human, viser_vis_world4d
from prompt_hmr.vis.traj import get_floor_mesh
from pipeline import Pipeline


def main(input_video='data/examples/boxing_short.mp4',
         static_camera=False,
         run_viser=False,
         viser_total=1500,
         viser_subsample=1,
         output_folder=None,
         use_streaming=False,
         use_chunked_processing=False,
         chunk_size=300):
    print("starting demo_video.py")
    smplx = SMPLX_Layer(SMPLX_PATH).cuda()
    if output_folder is None:
        output_folder = 'results/' + \
            os.path.basename(input_video).split('.')[0]
    else:
        output_folder = output_folder

    pipeline = Pipeline(
        static_cam=static_camera,
        use_streaming=use_streaming,
        use_chunked_processing=use_chunked_processing,
        chunk_size=chunk_size
    )
    results = pipeline.__call__(input_video,
                                output_folder,
                                save_only_essential=True)


if __name__ == '__main__':
    tyro.cli(main)