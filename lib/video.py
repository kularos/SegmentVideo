#!/usr/bin/env python3
"""
Video loader for assisted manual segmentation.
Loads video into (3, W, H, F) tensor format.
"""

import numpy as np
import cv2
from pathlib import Path


def load_video_to_tensor(video_path: str, pZ: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a video file into a (3, W, H, F) tensor.
    
    Args:
        video_path: Path to the video file
        pZ: Temporal zoom parameter (base-2 logarithm of frame skip).
            pZ=0: Load every frame (skip=1)
            pZ=1: Load every 2nd frame (skip=2)
            pZ=2: Load every 4th frame (skip=4)
            pZ=3: Load every 8th frame (skip=8)
            In general: skip = 2^pZ
        
    Returns:
        tuple of (tensor, frame_indices):
        - tensor: numpy array of shape (3, W, H, F) where:
            - 3: RGB channels (0-1 float range)
            - W: Width (left to right)
            - H: Height (bottom to top, so flipped from standard image coordinates)
            - F: Number of frames (temporal sequence, downsampled by 2^pZ)
        - frame_indices: numpy array of shape (F,) containing original frame indices
            Maps tensor frame index to original video frame index
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate temporal downsampling
    frame_skip = 2 ** pZ
    downsampled_frames = (total_frames + frame_skip - 1) // frame_skip  # Ceiling division
    effective_fps = fps / frame_skip
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Original frames: {total_frames}")
    print(f"  pZ (temporal zoom): {pZ}")
    print(f"  Frame skip: {frame_skip} (loading every {frame_skip}{'st' if frame_skip==1 else 'nd' if frame_skip==2 else 'rd' if frame_skip==3 else 'th'} frame)")
    print(f"  Downsampled frames: {downsampled_frames}")
    print(f"  Original FPS: {fps:.2f}")
    print(f"  Effective FPS: {effective_fps:.2f}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    
    # Calculate memory size
    memory_size_bytes = 3 * width * height * downsampled_frames * 4  # 4 bytes per float32
    memory_size_gb = memory_size_bytes / (1024**3)
    print(f"  Estimated tensor size: {memory_size_gb:.2f} GB")
    
    # Preallocate tensor (3, W, H, F)
    tensor = np.zeros((3, width, height, downsampled_frames), dtype=np.float32)
    
    # Track original frame indices
    frame_indices = np.zeros(downsampled_frames, dtype=np.int32)
    
    # Read frames
    frame_idx = 0
    output_frame_idx = 0
    print(f"\nLoading frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process frames that match our skip pattern
        if frame_idx % frame_skip == 0:
            # OpenCV reads as BGR, convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize to 0-1 range
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            
            # Flip vertically so height increases bottom to top
            frame_flipped = np.flip(frame_normalized, axis=0)
            
            # Reshape from (H, W, 3) to (3, W, H) and store
            # Note: transpose to get (3, W, H) from (H, W, 3)
            frame_transposed = np.transpose(frame_flipped, (2, 1, 0))
            tensor[:, :, :, output_frame_idx] = frame_transposed
            
            # Store original frame index
            frame_indices[output_frame_idx] = frame_idx
            
            output_frame_idx += 1
            if output_frame_idx % 100 == 0:
                print(f"  Loaded {output_frame_idx}/{downsampled_frames} frames ({100*output_frame_idx/downsampled_frames:.1f}%)")
        
        frame_idx += 1
    
    cap.release()
    
    print(f"\nSuccessfully loaded {output_frame_idx} frames (skipped {frame_idx - output_frame_idx})")
    print(f"Final tensor shape: {tensor.shape}")
    print(f"Frame indices range: {frame_indices[0]} to {frame_indices[-1]}")
    print(f"Actual memory usage: {tensor.nbytes / (1024**3):.2f} GB")
    
    return tensor, frame_indices


def get_video_info(video_path: str) -> dict:
    """
    Get video information without loading the entire video.
    Useful for checking memory requirements before loading.
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
    }
    
    cap.release()
    
    # Calculate memory requirements
    memory_bytes = 3 * info['width'] * info['height'] * info['frames'] * 4
    info['memory_gb'] = memory_bytes / (1024**3)
    info['duration_seconds'] = info['frames'] / info['fps'] if info['fps'] > 0 else 0
    
    return info


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_loader.py <video_file> [pZ]")
        print("\nArguments:")
        print("  video_file: Path to video file")
        print("  pZ: Temporal zoom (optional, default=0)")
        print("      pZ=0: Load every frame")
        print("      pZ=1: Load every 2nd frame")
        print("      pZ=2: Load every 4th frame")
        print("      pZ=3: Load every 8th frame")
        print("\nExample:")
        print("  python video_loader.py input.wmv 2")
        sys.exit(1)
    
    video_file = sys.argv[1]
    pZ = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    # First check video info
    print("Checking video information...")
    info = get_video_info(video_file)
    
    # Calculate downsampled memory
    frame_skip = 2 ** pZ
    downsampled_frames = (info['frames'] + frame_skip - 1) // frame_skip
    downsampled_memory_gb = (3 * info['width'] * info['height'] * downsampled_frames * 4) / (1024**3)
    
    print(f"\nOriginal: {info['frames']} frames, {info['memory_gb']:.2f} GB")
    print(f"With pZ={pZ}: {downsampled_frames} frames, {downsampled_memory_gb:.2f} GB")
    print(f"Memory reduction: {100 * (1 - downsampled_memory_gb/info['memory_gb']):.1f}%")
    
    response = input("\nProceed with loading? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Load the video
    tensor, frame_indices = load_video_to_tensor(video_file, pZ=pZ)
    
    # Example: Access a specific frame
    # frame_10 = tensor[:, :, :, 10]  # (3, W, H) for frame 10
    # original_frame_idx = frame_indices[10]  # Original frame number in video
    
    print("\nTensor loaded successfully!")
    print("You can now use the tensor for your segmentation algorithm.")
    print(f"Frame indices: {frame_indices[:5]}... (first 5 shown)")
