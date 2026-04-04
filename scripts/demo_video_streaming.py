#!/usr/bin/env python3
"""
Simple streaming video processing script.
Handles long videos efficiently without OOM.

Usage:
    python scripts/demo_video_streaming.py --input_video long_video.mp4
"""

import os
import sys
import tyro
import torch

sys.path.insert(0, os.path.dirname(__file__) + '/..')

from data_config import SMPLX_PATH
from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from pipeline import Pipeline


def main(
    input_video: str = 'data/examples/boxing_short.mp4',
    output_folder: str = None,
    static_camera: bool = True,
    use_streaming: bool = True,
    use_chunked_processing: bool = False,
    chunk_size: int = 300,
):
    """
    Process video with memory optimization.
    
    Args:
        input_video: Path to input video
        output_folder: Output directory (auto-generated if not specified)
        static_camera: Assume static camera (faster)
        use_streaming: Load frames on-demand instead of all at once
        use_chunked_processing: Process in chunks for very long videos
        chunk_size: Frames per chunk (reduce for tighter memory)
    """
    print("="*60)
    print("PromptHMR - Memory Optimized Video Processing")
    print("="*60)
    
    # Auto-generate output folder
    if output_folder is None:
        base_name = os.path.basename(input_video).split('.')[0]
        output_folder = f'results/{base_name}'

    print(f"\n📹 Input: {input_video}")
    print(f"💾 Output: {output_folder}")
    print(f"⚙️  Options:")
    print(f"   - Static camera: {static_camera}")
    print(f"   - Streaming: {use_streaming}")
    print(f"   - Chunked: {use_chunked_processing}")
    if use_chunked_processing:
        print(f"   - Chunk size: {chunk_size} frames")
    
    print("\n" + "="*60)
    
    # Initialize SMPLX
    print("Loading SMPLX model...")
    smplx = SMPLX_Layer(SMPLX_PATH).cuda()
    
    # Create pipeline
    print("Initializing pipeline...")
    pipeline = Pipeline(
        static_cam=static_camera,
        use_streaming=use_streaming,
        use_chunked_processing=use_chunked_processing,
        chunk_size=chunk_size
    )
    
    # Run processing
    print("\nStarting processing...")
    print("-"*60)
    
    try:
        results = pipeline(
            input_video,
            output_folder,
            save_only_essential=True
        )
        
        print("-"*60)
        print("\n✅ Processing completed successfully!")
        print(f"📊 Results saved to: {output_folder}/results.pkl")
        print(f"🎬 MCS visualization: {output_folder}/world4d.mcs")
        
    except torch.cuda.OutOfMemoryError:
        print("\n❌ GPU Out of Memory!")
        print("💡 Try these solutions:")
        print("   1. Increase --chunk_size value")
        print("   2. Reduce input resolution (modify max_height in pipeline)")
        print("   3. Use smaller batch size (modify in pipeline/phmr_vid.py)")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    tyro.cli(main)
