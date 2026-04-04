"""
Streaming video dataset for memory-efficient processing of long videos.
Instead of loading all frames into memory, frames are read on-demand from disk.
"""

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import Normalize, ToTensor, Compose
from pathlib import Path
import os


class StreamingVideoDataset:
    """Memory-efficient video dataset that reads frames from disk on-demand."""
    
    def __init__(self, image_paths, video_path=None, max_memory_frames=32):
        """
        Initialize streaming dataset.
        
        Args:
            image_paths: List of image file paths OR path to video file
            video_path: Optional path to video file if image_paths is not provided
            max_memory_frames: Maximum number of frames to keep in memory cache
        """
        self.max_memory_frames = max_memory_frames
        self.frame_cache = {}  # Simple LRU cache
        self.cache_order = []  # Track insertion order for LRU
        
        # Handle both image list and video file
        
        if isinstance(image_paths, list):
            self.image_paths = sorted(image_paths)
        else:
            raise ValueError(
                "image_paths must be a list of paths or a video file path")
    
    
    def __len__(self):
        """Return total number of frames."""
        return len(self.image_paths) if self.image_paths else 0
    
    def _add_to_cache(self, frame_idx, frame):
        """Add frame to cache with LRU eviction policy."""
        self.frame_cache[frame_idx] = frame
        self.cache_order.append(frame_idx)
        
        # Remove oldest frame if cache is full
        if len(self.frame_cache) > self.max_memory_frames:
            oldest_idx = self.cache_order.pop(0)
            del self.frame_cache[oldest_idx]
    
    def get_frame(self, frame_idx):
        """
        Get frame at index, using cache when available.
        
        Args:
            frame_idx: Index of frame to retrieve
            
        Returns:
            numpy array of frame in RGB format
        """
        # Check cache first
        if frame_idx in self.frame_cache:
            return self.frame_cache[frame_idx]

        # Load from source
        
        frame = self._get_frame_from_image(frame_idx)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Add to cache
        self._add_to_cache(frame_idx, frame)
        
        return frame
    
    def _get_frame_from_image(self, frame_idx):
        """Load frame from image file."""
        if frame_idx < 0 or frame_idx >= len(self.image_paths):
            raise IndexError(f"Frame index {frame_idx} out of range")
        frame = cv2.imread(self.image_paths[frame_idx])
        if frame is None:
            raise RuntimeError(f"Failed to read image: {self.image_paths[frame_idx]}")
        return frame
    
    def get_batch_frames(self, frame_indices):
        """
        Get multiple frames efficiently.
        
        Args:
            frame_indices: List of frame indices
            
        Returns:
            numpy array of shape (N, H, W, 3)
        """
        frames = [self.get_frame(idx) for idx in frame_indices]
        return np.stack(frames, axis=0)
    
    def clear_cache(self):
        """Clear frame cache to free memory."""
        self.frame_cache.clear()
        self.cache_order.clear()


class PromptHMRVideoDatasetStreaming:
    """Streaming version of PromptHMRVideoDataset that doesn't store masks in memory."""
    
    def __init__(self, streaming_dataset, tracks, cam_int, save_masks_to_disk=True, 
                 mask_cache_dir=None):
        """
        Initialize streaming dataset.
        
        Args:
            streaming_dataset: StreamingVideoDataset instance
            tracks: Dictionary of tracks with bounding boxes and keypoints
            cam_int: Camera intrinsic matrix
            save_masks_to_disk: Whether to save masks to disk instead of memory
            mask_cache_dir: Directory to cache masks (if save_masks_to_disk=True)
        """
        self.streaming_dataset = streaming_dataset
        self.tracks = tracks
        self.cam_int = cam_int
        self.save_masks_to_disk = save_masks_to_disk
        
        # Setup mask caching
        if save_masks_to_disk and mask_cache_dir:
            self.mask_cache_dir = mask_cache_dir
            os.makedirs(mask_cache_dir, exist_ok=True)
        else:
            self.mask_cache_dir = None
        
        frames = set([x for t in tracks.values()
                     for x in t['frames'].tolist()])
        print("total frames:", len(frames))
        print("frames:", frames)
        if (len(frames) == 0):
            print(tracks)

        self.frames = sorted(list(frames))
        
        self.normalization = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], 
                      std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        """Get item without loading all masks into memory."""
        frame_idx = self.frames[idx]
        image_cv = self.streaming_dataset.get_frame(frame_idx)
        
        boxes = []
        kpts = []
        mask_paths = []  # Instead of storing masks, store paths
        track_ids = []
        
        for person_id, track in self.tracks.items():
            if frame_idx in track['frames']:
                bbox = track['bboxes'][track['frames'] == frame_idx]
                if bbox.ndim == 1:
                    bbox = bbox[None]
                bbox = np.concatenate([bbox, np.ones_like(bbox)[..., :1]], axis=-1)
                
                kpt = track['keypoints_2d'][track['frames'] == frame_idx][:, :25]
                
                # Handle masks efficiently
                if self.save_masks_to_disk and self.mask_cache_dir:
                    # Save masks to disk instead of keeping in memory
                    mask_path = self._save_or_get_mask_path(person_id, frame_idx, track)
                    mask_paths.append(mask_path)
                else:
                    # If masks are already in memory, use them
                    if 'masks' in track:
                        mask_paths.append(None)  # Will load from track on-demand
                    else:
                        mask_paths.append(None)
                
                track_ids.append(person_id)
                boxes.append(bbox)
                kpts.append(kpt)
        
        boxes = torch.from_numpy(np.concatenate(boxes)).float()
        kpts = torch.from_numpy(np.concatenate(kpts)).float() if len(kpts) > 0 else torch.empty(0, 25, 3)
        
        # Load masks only when needed (will be handled in batch processing)
        cam_int_batch = self.cam_int.float()[None].repeat(len(boxes), 1, 1)
        
        item = {
            'boxes': boxes,
            'cam_int': cam_int_batch,
            'image_cv': image_cv,
            'track_ids': track_ids,
            'kpts': kpts,
            'mask_paths': mask_paths,  # Store paths instead of actual masks
            'frame_idx': frame_idx,
            'tracks_ref': self.tracks,  # Reference to tracks for on-demand mask loading
        }
        
        item = self._pad_image(item, IMG_SIZE=896)
        item['image'] = self.normalization(item['image_cv'])
        item['image_cv'] = torch.tensor(item['image_cv'])
        
        return item
    
    def _save_or_get_mask_path(self, person_id, frame_idx, track):
        """Save mask to disk or get existing path."""
        mask_file = os.path.join(
            self.mask_cache_dir, 
            f"mask_p{person_id}_f{frame_idx}.npz"
        )
        
        if not os.path.exists(mask_file):
            if 'masks' in track:
                track_frames = np.atleast_1d(track['frames'].tolist())
                mask_frame_idx = np.where(track_frames == frame_idx)[0]
                if len(mask_frame_idx) > 0:
                    mask = track['masks'][mask_frame_idx[0]]
                    np.savez_compressed(mask_file, mask=mask)
        
        return mask_file
    
    @staticmethod
    def _pad_image(item, IMG_SIZE=896):
        """Pad image to fixed size."""
        img = item['image_cv']
        size = np.array([img.shape[1], img.shape[0]])
        scale = IMG_SIZE / max(size)
        offset = (IMG_SIZE - scale * size) / 2
        
        img_pil = Image.fromarray(img)
        img_pil = ImageOps.contain(img_pil, (IMG_SIZE, IMG_SIZE))
        img_pil = ImageOps.pad(img_pil, size=(IMG_SIZE, IMG_SIZE))
        img = np.array(img_pil)
        
        item['image_cv'] = img
        item['cam_int'] = item['cam_int'].mean(dim=0, keepdim=True)
        item['cam_int'][:, :2] *= scale
        item['cam_int'][:, :2, -1] += offset
        item['boxes'] *= scale
        item['boxes'][:, :2] += offset
        item['boxes'][:, 2:4] += offset
        
        kpts = item.get('kpts', None)
        if kpts is not None and kpts.numel() > 0:
            # Ensure kpts is 3D: (N, num_joints, 3)
            if kpts.dim() == 4:
                kpts = kpts.reshape(kpts.shape[0], -1, kpts.shape[-1])
            if kpts.shape[-1] == 3:
                kpts[:, :, :2] *= scale
                kpts[:, :, :2] += offset
                item['kpts'] = kpts
        
        return item
    
    def load_masks_for_batch(self, batch_items):
        """Load masks on-demand for a batch of items."""
        masks_list = []
        
        for item in batch_items:
            batch_masks = []
            for i, track_id in enumerate(item['track_ids']):
                track = item['tracks_ref'][track_id]
                frame_idx = item['frame_idx']
                
                track_frames = np.atleast_1d(track['frames'])
                mask_frame_idx = np.where(track_frames == frame_idx)[0]
                if len(mask_frame_idx) > 0 and 'masks' in track:
                    obj_masks = track['masks'][mask_frame_idx]
                    
                    mm = []
                    for mask in obj_masks:
                        msk_size = int(896 / 14 * 4)
                        mask_img = Image.fromarray(mask)
                        mask_img = ImageOps.contain(mask_img, (msk_size, msk_size))
                        mask_img = ImageOps.pad(mask_img, size=(msk_size, msk_size))
                        mm.append(np.array(mask_img))
                    
                    mask_tensor = torch.from_numpy(np.array(mm)).float()
                else:
                    # Return empty mask if not available
                    msk_size = int(896 / 14 * 4)
                    mask_tensor = torch.ones(1, msk_size, msk_size).float()
                
                batch_masks.append(mask_tensor)
            
            # Concatenate all masks for this item
            if batch_masks:
                item_masks = torch.cat(batch_masks, dim=0)[:, None]
            else:
                msk_size = int(896 / 14 * 4)
                item_masks = torch.ones(1, 1, msk_size, msk_size).float()
            
            masks_list.append(item_masks)
        
        # Stack masks for the batch
        if masks_list:
            batch_masks_tensor = torch.cat(masks_list, dim=0)
        else:
            msk_size = int(896 / 14 * 4)
            batch_masks_tensor = torch.ones(1, 1, msk_size, msk_size).float()
        
        return batch_masks_tensor
