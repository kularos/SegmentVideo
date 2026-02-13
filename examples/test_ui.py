#!/usr/bin/env python3
"""
Simple test script to demonstrate the interactive verification UI.
Creates synthetic frames and shows the verification interface.
"""

import numpy as np
from lib.annotation import (
    create_ellipsoid_params, 
    create_chain_params,
    verify_with_ui, 
    AnnotationStatus
)


def create_test_frame(width=640, height=480):
    """
    Create a synthetic test frame with some visual features.
    
    Returns:
        (3, W, H) tensor with RGB gradient pattern
    """
    # Create a gradient pattern
    frame = np.zeros((3, width, height), dtype=np.float32)
    
    # Red channel: left-right gradient
    frame[0, :, :] = np.linspace(0, 1, width)[:, np.newaxis]
    
    # Green channel: bottom-top gradient (remember our coordinate system!)
    frame[1, :, :] = np.linspace(0, 1, height)[np.newaxis, :]
    
    # Blue channel: radial gradient from center
    x = np.linspace(0, 1, width)[:, np.newaxis]
    y = np.linspace(0, 1, height)[np.newaxis, :]
    cx, cy = 0.5, 0.5
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    frame[2, :, :] = 1 - np.clip(dist / 0.7, 0, 1)
    
    return frame


def test_verification_ui():
    """Test the interactive verification UI with a synthetic frame."""
    print("="*80)
    print("INTERACTIVE VERIFICATION UI TEST")
    print("="*80)
    print("\nGenerating synthetic test frame...")
    
    # Create test frame
    frame = create_test_frame(width=640, height=480)
    
    # Create a test prediction (ellipsoid in center)
    predicted_params = create_ellipsoid_params(
        frame_idx=0,
        center=np.array([320, 240, 100], dtype=np.float32),
        semi_axes=np.array([80, 60, 40], dtype=np.float32),
        rotation=np.array([0, 0, 0.3], dtype=np.float32),
        status=AnnotationStatus.MODEL_PREDICTED
    )
    predicted_params.confidence = 0.85
    
    print("\nTest parameters:")
    print(f"  Frame size: {frame.shape[1]} x {frame.shape[2]}")
    print(f"  Model: {predicted_params.model_type}")
    print(f"  Center: {predicted_params.parameters['center']}")
    print(f"  Semi-axes: {predicted_params.parameters['semi_axes']}")
    print(f"  Rotation (yaw): {predicted_params.parameters['rotation'][2]:.2f} rad")
    
    print("\n" + "-"*80)
    print("Opening interactive verification window...")
    print("Instructions:")
    print("  - Press 'A' or click 'Accept' to accept the prediction")
    print("  - Press 'S' or click 'Skip' to skip")
    print("  - Modify parameters in text boxes and press Enter to update")
    print("-"*80)
    
    # Show verification UI
    accepted, final_params = verify_with_ui(
        frame, 
        predicted_params,
        frame_idx=0,
        tensor_idx=0,
        total_frames=1
    )
    
    print("\n" + "="*80)
    print("VERIFICATION RESULT")
    print("="*80)
    
    if accepted:
        print("✓ User ACCEPTED the annotation")
        print("\nFinal parameters:")
        print(f"  Center: {final_params.parameters['center']}")
        print(f"  Semi-axes: {final_params.parameters['semi_axes']}")
        print(f"  Rotation (yaw): {final_params.parameters['rotation'][2]:.2f} rad")
        
        # Check if parameters were modified
        center_changed = not np.allclose(
            predicted_params.parameters['center'], 
            final_params.parameters['center']
        )
        axes_changed = not np.allclose(
            predicted_params.parameters['semi_axes'], 
            final_params.parameters['semi_axes']
        )
        
        if center_changed or axes_changed:
            print("\n  → Parameters were MODIFIED by user")
        else:
            print("\n  → Parameters were ACCEPTED as-is")
    else:
        print("✗ User SKIPPED this frame")
    
    print("="*80)


def test_chain_ui():
    """Test the interactive verification UI with a chain/curve."""
    print("="*80)
    print("CHAIN/CURVE VERIFICATION UI TEST")
    print("="*80)
    print("\nGenerating synthetic test frame...")
    
    # Create test frame
    frame = create_test_frame(width=640, height=480)
    
    # Create a test chain with 5 segments (6 points)
    predicted_params = create_chain_params(
        frame_idx=0,
        start_point=np.array([100, 100, 0], dtype=np.float32),
        end_point=np.array([540, 380, 0], dtype=np.float32),
        n_segments=5,
        status=AnnotationStatus.MODEL_PREDICTED
    )
    predicted_params.confidence = 0.82
    
    print("\nTest parameters:")
    print(f"  Frame size: {frame.shape[1]} x {frame.shape[2]}")
    print(f"  Model: {predicted_params.model_type}")
    print(f"  Number of points: {len(predicted_params.parameters['points'])}")
    print(f"  Start point: {predicted_params.parameters['points'][0]}")
    print(f"  End point: {predicted_params.parameters['points'][-1]}")
    
    print("\n" + "-"*80)
    print("Opening interactive verification window...")
    print("Instructions:")
    print("  - Drag ENDPOINTS (red squares) freely to adjust chain")
    print("  - Drag MIDDLE POINTS (red circles) perpendicular to backbone only")
    print("  - Spacing between points stays constant (evenly distributed)")
    print("  - Press 'A' or click 'Accept' to accept")
    print("  - Press 'S' or click 'Skip' to skip")
    print("-"*80)
    
    # Show verification UI
    accepted, final_params = verify_with_ui(
        frame,
        predicted_params,
        frame_idx=0,
        tensor_idx=0,
        total_frames=1
    )
    
    print("\n" + "="*80)
    print("VERIFICATION RESULT")
    print("="*80)
    
    if accepted:
        print("✓ User ACCEPTED the annotation")
        print("\nFinal chain points:")
        for i, point in enumerate(final_params.parameters['points']):
            print(f"  Point {i}: [{point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}]")
        
        # Check if points were modified
        points_changed = not np.allclose(
            predicted_params.parameters['points'],
            final_params.parameters['points']
        )
        
        if points_changed:
            print("\n  → Chain was MODIFIED by user (points dragged)")
        else:
            print("\n  → Chain was ACCEPTED as-is")
    else:
        print("✗ User SKIPPED this frame")
    
    print("="*80)


if __name__ == "__main__":

    print("This script demonstrates the interactive verification UI.")
    print("Choose which test to run:\n")
    print("  [1] Ellipsoid test (drag center and axis)")
    print("  [2] Chain/curve test (drag control points)")
    print("  [3] Both tests")
    
    choice = input("\nSelect test (1-3) or press Enter for both: ").strip()
    
    try:
        if choice == '' or choice == '3':
            print("\n" + "="*80)
            print("RUNNING ELLIPSOID TEST")
            print("="*80)
            test_verification_ui()
            
            print("\n\n" + "="*80)
            print("RUNNING CHAIN TEST")
            print("="*80)
            test_chain_ui()
        elif choice == '1':
            test_verification_ui()
        elif choice == '2':
            test_chain_ui()
        else:
            print("Invalid choice. Running both tests.")
            test_verification_ui()
            test_chain_ui()
        
        print("\nAll tests completed successfully!")
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\nError during test: {e}")
        import traceback
        traceback.print_exc()
