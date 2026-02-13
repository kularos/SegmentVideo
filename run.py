#!/usr/bin/env python3
"""
Run script for assisted manual segmentation.
Automatically finds videos in ./videos directory and lets user select one.
"""

import sys
from pathlib import Path


def find_videos(directory="./videos"):
    """
    Find all video files in the specified directory.
    
    Args:
        directory: Path to directory containing videos
        
    Returns:
        List of video file paths
    """
    video_dir = Path(directory)
    
    if not video_dir.exists():
        return []
    
    # Common video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.mkv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg'}
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f'*{ext}'))
        video_files.extend(video_dir.glob(f'*{ext.upper()}'))
    
    return sorted(video_files)


def select_video(video_files):
    """
    Display menu for user to select a video.
    
    Args:
        video_files: List of Path objects
        
    Returns:
        Selected video path or None
    """
    if not video_files:
        return None
    
    print("\n" + "="*80)
    print("AVAILABLE VIDEOS")
    print("="*80)
    
    for i, video in enumerate(video_files, 1):
        # Get file size
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {video.name} ({size_mb:.1f} MB)")
    
    print("="*80)
    
    while True:
        try:
            choice = input(f"\nSelect video (1-{len(video_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(video_files):
                return video_files[idx]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(video_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")


def select_pz():
    """
    Prompt user to select pZ value.
    
    Returns:
        pZ value (int)
    """
    print("\n" + "="*80)
    print("TEMPORAL ZOOM LEVEL (pZ)")
    print("="*80)
    print("pZ controls temporal downsampling (frame skip = 2^pZ):")
    print("  pZ=0: Load every frame (no downsampling)")
    print("  pZ=1: Load every 2nd frame")
    print("  pZ=2: Load every 4th frame (recommended starting point)")
    print("  pZ=3: Load every 8th frame (coarse, low memory)")
    print("  pZ=4: Load every 16th frame (very coarse)")
    print("="*80)
    
    while True:
        try:
            choice = input("\nSelect pZ (0-4) or press Enter for default (2): ").strip()
            
            if choice == '':
                return 2
            
            pz = int(choice)
            if 0 <= pz <= 4:
                return pz
            else:
                print("Invalid choice. Please enter a number between 0 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number or press Enter for default.")


def select_mode():
    """
    Prompt user to select interactive mode.
    
    Returns:
        True for interactive UI, False for auto-accept
    """
    print("\n" + "="*80)
    print("VERIFICATION MODE")
    print("="*80)
    print("  [1] Interactive UI - Verify each keyframe with matplotlib window (recommended)")
    print("  [2] Auto-accept - Automatically accept all predictions (for testing)")
    print("="*80)
    
    while True:
        choice = input("\nSelect mode (1-2) or press Enter for interactive (1): ").strip()
        
        if choice == '' or choice == '1':
            return True
        elif choice == '2':
            return False
        else:
            print("Invalid choice. Please enter 1 or 2.")


def main():
    """Main run script."""
    print("="*80)
    print("ASSISTED MANUAL SEGMENTATION - RUN SCRIPT")
    print("="*80)
    
    # Check if videos directory exists
    videos_dir = Path("./videos")
    if not videos_dir.exists():
        print(f"\n⚠ Videos directory not found: {videos_dir.absolute()}")
        print("\nCreating ./videos directory...")
        videos_dir.mkdir(parents=True)
        print(f"✓ Created {videos_dir.absolute()}")
        print("\nPlease add video files to this directory and run the script again.")
        return 1
    
    # Find video files
    video_files = find_videos("./videos")
    
    if not video_files:
        print(f"\n⚠ No video files found in {videos_dir.absolute()}")
        print("\nSupported formats: .mp4, .avi, .mov, .wmv, .mkv, .flv, .webm, .m4v, .mpg, .mpeg")
        print("\nPlease add video files to the ./videos directory and run again.")
        return 1
    
    # User selects video
    selected_video = select_video(video_files)
    if selected_video is None:
        print("\nOperation cancelled by user.")
        return 0
    
    print(f"\n✓ Selected: {selected_video.name}")
    
    # User selects pZ
    pz = select_pz()
    print(f"\n✓ Selected pZ={pz} (every {2**pz} frames)")
    
    # User selects mode
    use_ui = select_mode()
    print(f"\n✓ Selected mode: {'Interactive UI' if use_ui else 'Auto-accept'}")
    
    # Confirm before proceeding
    print("\n" + "="*80)
    print("READY TO START")
    print("="*80)
    print(f"Video: {selected_video.name}")
    print(f"pZ: {pz} (every {2**pz} frames)")
    print(f"Mode: {'Interactive UI' if use_ui else 'Auto-accept'}")
    print("="*80)
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("\nOperation cancelled.")
        return 0
    
    # Import and run workflow
    print("\n" + "="*80)
    print("STARTING WORKFLOW")
    print("="*80 + "\n")
    
    try:
        from examples.example_workflow import assisted_segmentation_workflow


        state, tensor, frame_indices = assisted_segmentation_workflow(
            str(selected_video),
            pZ=pz,
            use_interactive_ui=use_ui
        )
        
        print("\n" + "="*80)
        print("WORKFLOW COMPLETE!")
        print("="*80)
        print(f"\nAnnotations for {selected_video.name} are ready.")
        print("You can now export or further process the results.")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ Workflow interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\n✗ Error during workflow: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
