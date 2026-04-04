"""
Chunked processing utilities for handling very long videos.
Processes videos in chunks to avoid OOM issues.
"""

import os
import numpy as np
import torch
import joblib
from typing import Dict, List, Any, Optional


class ChunkedVideoProcessor:
    """Process long videos in chunks to manage memory efficiently."""
    
    def __init__(self, chunk_size=300, overlap=30):
        """
        Initialize chunked processor.
        
        Args:
            chunk_size: Number of frames per chunk
            overlap: Number of overlapping frames between chunks for continuity
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def get_chunk_indices(self, total_frames):
        """
        Get frame indices for each chunk.
        
        Args:
            total_frames: Total number of frames in video
            
        Returns:
            List of (start, end) tuples for each chunk
        """
        chunks = []
        start = 0
        
        while start < total_frames:
            end = min(start + self.chunk_size, total_frames)
            chunks.append((start, end))
            
            # Move start position for next chunk (with overlap)
            if end < total_frames:
                start = end - self.overlap
            else:
                break
        
        return chunks
    
    def merge_track_results(self, chunk_results: List[Dict], 
                           overlap_indices: List[tuple]) -> Dict:
        """
        Merge results from multiple chunks, handling overlaps intelligently.
        
        Args:
            chunk_results: List of results from each chunk
            overlap_indices: List of (start, end) overlap indices for each chunk
            
        Returns:
            Merged results dictionary
        """
        if not chunk_results:
            return {}
        
        merged = {}
        
        for person_id in chunk_results[0].get('people', {}):
            merged[person_id] = self._merge_person_tracks(
                chunk_results, person_id, overlap_indices
            )
        
        return merged
    
    def _merge_person_tracks(self, chunk_results: List[Dict], 
                            person_id: int, overlap_indices: List[tuple]) -> Dict:
        """Merge single person's tracks across chunks."""
        merged_track = {
            'frames': [],
            'bboxes': [],
            'keypoints_2d': [],
            'vitpose': [],
            'smplx_pose': [],
            'smplx_transl': [],
            'smplx_betas': [],
            'prhmr_img_feats': [],
        }
        
        for chunk_idx, result in enumerate(chunk_results):
            if person_id not in result.get('people', {}):
                continue
            
            track = result['people'][person_id]
            
            # Handle overlap: skip overlapping frames in subsequent chunks
            if chunk_idx > 0:
                overlap_start, overlap_end = overlap_indices[chunk_idx]
                # Skip frames that are in the overlap region of previous chunk
                keep_from = overlap_end - overlap_start[0]
            else:
                keep_from = 0
            
            # Merge frame indices (adjust for chunk offset)
            chunk_start = overlap_indices[chunk_idx][0]
            frames = track.get('frames', np.array([])) + chunk_start
            
            if keep_from > 0:
                for key in merged_track:
                    if key in track:
                        if isinstance(track[key], np.ndarray):
                            merged_track[key].append(track[key][keep_from:])
                        elif isinstance(track[key], torch.Tensor):
                            merged_track[key].append(track[key][keep_from:])
                        elif isinstance(track[key], list):
                            merged_track[key].extend(track[key][keep_from:])
            else:
                for key in merged_track:
                    if key in track:
                        if isinstance(track[key], list):
                            merged_track[key].extend(track[key])
                        else:
                            merged_track[key].append(track[key])
        
        # Stack or concatenate results
        for key in merged_track:
            if merged_track[key]:
                if isinstance(merged_track[key][0], np.ndarray):
                    merged_track[key] = np.concatenate(merged_track[key], axis=0)
                elif isinstance(merged_track[key][0], torch.Tensor):
                    merged_track[key] = torch.cat(merged_track[key], dim=0)
            else:
                del merged_track[key]
        
        return merged_track


def save_intermediate_results(results: Dict, output_folder: str, chunk_idx: int):
    """Save intermediate results for a chunk."""
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f'chunk_{chunk_idx:03d}.pkl')
    joblib.dump(results, filename)
    return filename


def load_intermediate_results(filename: str) -> Dict:
    """Load intermediate results from a chunk."""
    return joblib.load(filename)


def cleanup_intermediate_results(output_folder: str):
    """Clean up intermediate chunk results."""
    import glob
    files = glob.glob(os.path.join(output_folder, 'chunk_*.pkl'))
    for f in files:
        os.remove(f)
