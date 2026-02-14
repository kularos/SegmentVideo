#!/usr/bin/env python3
"""
Example: Using the EllipsoidModel for tracking.

This demonstrates:
1. Creating ellipsoid models
2. Dragging control points
3. Interpolation between ellipsoids
4. Rendering
"""

import numpy as np
import matplotlib.pyplot as plt
from segmentvideo.models import EllipsoidModel


def example_ellipsoid_basic():
    """Basic ellipsoid creation and rendering."""
    print("=" * 60)
    print("Example 1: Basic Ellipsoid Model")
    print("=" * 60)
    
    # Create an ellipsoid from 2D parameters
    ellipsoid = EllipsoidModel.from_2d(
        frame_idx=0,
        center_x=320,
        center_y=240,
        width=80,
        height=50,
        angle=np.pi / 6  # 30 degrees
    )
    
    print(f"\nEllipsoid created:")
    print(f"  Center: {ellipsoid.center}")
    print(f"  Semi-axes: {ellipsoid.semi_axes}")
    print(f"  Rotation: {ellipsoid.rotation} radians")
    print(f"  Area: {ellipsoid.get_area():.1f} pixels²")
    print(f"  Volume: {ellipsoid.get_volume():.1f} pixels³")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create a simple background
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Ellipsoid Model - Basic Example")
    
    # Render ellipsoid
    ellipsoid.render(ax, color='red', linewidth=2)
    
    # Get and display control points
    control_points = ellipsoid.get_control_points()
    print(f"\nControl points:")
    for i, cp in enumerate(control_points):
        print(f"  Point {i} ({cp.point_type}): ({cp.x:.1f}, {cp.y:.1f})")
    
    plt.show()


def example_ellipsoid_dragging():
    """Demonstrate dragging control points."""
    print("\n" + "=" * 60)
    print("Example 2: Dragging Control Points")
    print("=" * 60)
    
    # Create initial ellipsoid
    ellipsoid = EllipsoidModel.from_2d(
        frame_idx=0,
        center_x=200,
        center_y=200,
        width=60,
        height=40,
        angle=0
    )
    
    print("\nDemonstrating control point updates:")
    print("Initial state:")
    print(f"  Center: ({ellipsoid.center[0]:.1f}, {ellipsoid.center[1]:.1f})")
    print(f"  Rotation: {np.degrees(ellipsoid.rotation[2]):.1f}°")
    
    # Drag center
    print("\n1. Dragging center to (300, 250)...")
    ellipsoid.update_from_control_point(0, 300, 250)
    print(f"  New center: ({ellipsoid.center[0]:.1f}, {ellipsoid.center[1]:.1f})")
    
    # Drag axis endpoint (rotates and scales)
    print("\n2. Dragging axis endpoint to (380, 280)...")
    ellipsoid.update_from_control_point(1, 380, 280)
    print(f"  New semi-axis a: {ellipsoid.semi_axes[0]:.1f}")
    print(f"  New rotation: {np.degrees(ellipsoid.rotation[2]):.1f}°")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original
    ellipsoid_orig = EllipsoidModel.from_2d(0, 200, 200, 60, 40, 0)
    ax1.set_xlim(0, 400)
    ax1.set_ylim(0, 400)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Original Ellipsoid")
    ellipsoid_orig.render(ax1, color='blue', linewidth=2)
    
    # After dragging
    ax2.set_xlim(0, 400)
    ax2.set_ylim(0, 400)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title("After Dragging Control Points")
    ellipsoid.render(ax2, color='red', linewidth=2)
    
    plt.tight_layout()
    plt.show()


def example_ellipsoid_interpolation():
    """Demonstrate interpolation between ellipsoids."""
    print("\n" + "=" * 60)
    print("Example 3: Interpolation Between Ellipsoids")
    print("=" * 60)
    
    # Create two ellipsoids
    ellipsoid1 = EllipsoidModel.from_2d(
        frame_idx=0,
        center_x=150,
        center_y=200,
        width=60,
        height=40,
        angle=0
    )
    
    ellipsoid2 = EllipsoidModel.from_2d(
        frame_idx=10,
        center_x=450,
        center_y=300,
        width=80,
        height=30,
        angle=np.pi / 3  # 60 degrees
    )
    
    print(f"\nEllipsoid 1 (frame 0):")
    print(f"  Center: ({ellipsoid1.center[0]:.1f}, {ellipsoid1.center[1]:.1f})")
    print(f"  Size: {ellipsoid1.semi_axes[0]:.1f} x {ellipsoid1.semi_axes[1]:.1f}")
    print(f"  Angle: {np.degrees(ellipsoid1.rotation[2]):.1f}°")
    
    print(f"\nEllipsoid 2 (frame 10):")
    print(f"  Center: ({ellipsoid2.center[0]:.1f}, {ellipsoid2.center[1]:.1f})")
    print(f"  Size: {ellipsoid2.semi_axes[0]:.1f} x {ellipsoid2.semi_axes[1]:.1f}")
    print(f"  Angle: {np.degrees(ellipsoid2.rotation[2]):.1f}°")
    
    # Create interpolated ellipsoids
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for i, alpha in enumerate(alphas):
        ax = axes[i]
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 500)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if alpha == 0.0:
            # Original ellipsoid 1
            ellipsoid1.render(ax, color='blue', linewidth=2)
            ax.set_title(f"Frame 0 (α={alpha:.1f})")
        elif alpha == 1.0:
            # Original ellipsoid 2
            ellipsoid2.render(ax, color='red', linewidth=2)
            ax.set_title(f"Frame 10 (α={alpha:.1f})")
        else:
            # Interpolated
            interpolated = ellipsoid1.interpolate(ellipsoid2, alpha)
            interpolated.render(ax, color='purple', linewidth=2)
            frame_idx = int((1 - alpha) * 0 + alpha * 10)
            ax.set_title(f"Frame {frame_idx} (α={alpha:.1f})")
            
            print(f"\nInterpolated at α={alpha:.1f}:")
            print(f"  Center: ({interpolated.center[0]:.1f}, {interpolated.center[1]:.1f})")
    
    plt.tight_layout()
    plt.show()


def example_ellipsoid_tracking():
    """Simulate a tracking scenario."""
    print("\n" + "=" * 60)
    print("Example 4: Simulated Tracking Scenario")
    print("=" * 60)
    
    # Simulate tracking an object that moves and rotates
    ellipsoids = []
    
    # Create a sequence of ellipsoids simulating motion
    for frame in range(11):
        t = frame / 10.0  # Normalized time [0, 1]
        
        # Circular motion
        angle = 2 * np.pi * t
        center_x = 300 + 150 * np.cos(angle)
        center_y = 300 + 150 * np.sin(angle)
        
        # Varying size (pulsating)
        size_factor = 1.0 + 0.3 * np.sin(4 * np.pi * t)
        width = 50 * size_factor
        height = 30 * size_factor
        
        # Rotating
        rotation_angle = np.pi * t
        
        ellipsoid = EllipsoidModel.from_2d(
            frame_idx=frame,
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
            angle=rotation_angle
        )
        ellipsoids.append(ellipsoid)
    
    print(f"\nCreated {len(ellipsoids)} ellipsoids simulating motion")
    
    # Visualize tracking
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 600)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Ellipsoid Tracking - Simulated Motion")
    
    # Draw all ellipsoids with varying transparency
    for i, ellipsoid in enumerate(ellipsoids):
        alpha = 0.3 + 0.7 * (i / len(ellipsoids))  # Fade from old to new
        color = 'blue' if i == 0 else 'red' if i == len(ellipsoids) - 1 else 'purple'
        ellipsoid.render(ax, color=color, linewidth=2, alpha=alpha, show_axes=(i == len(ellipsoids) - 1))
        
        # Draw trajectory line
        if i > 0:
            prev = ellipsoids[i-1]
            ax.plot([prev.center[0], ellipsoid.center[0]], 
                   [prev.center[1], ellipsoid.center[1]], 
                   'k--', alpha=0.3, linewidth=1)
    
    # Add labels
    ax.plot(ellipsoids[0].center[0], ellipsoids[0].center[1], 
           'bo', markersize=12, label='Start (Frame 0)')
    ax.plot(ellipsoids[-1].center[0], ellipsoids[-1].center[1], 
           'ro', markersize=12, label='End (Frame 10)')
    ax.legend()
    
    plt.show()


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("EllipsoidModel Examples")
    print("=" * 60)
    print("\nThis script demonstrates the EllipsoidModel class:")
    print("  1. Basic creation and rendering")
    print("  2. Dragging control points")
    print("  3. Interpolation between ellipsoids")
    print("  4. Simulated tracking scenario")
    print("\nClose each window to proceed to the next example.")
    print("=" * 60 + "\n")
    
    try:
        example_ellipsoid_basic()
        example_ellipsoid_dragging()
        example_ellipsoid_interpolation()
        example_ellipsoid_tracking()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
        sys.exit(0)
