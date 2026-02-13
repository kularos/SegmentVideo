#!/usr/bin/env python3
"""
Demonstration of complex number rotation for chain offsets.
Shows how perpendicular offsets rotate with the chain when endpoints are moved.
"""

import numpy as np
import matplotlib.pyplot as plt


def create_curved_chain(start, end, n_segments, curvature=0.3):
    """
    Create a chain with some curvature.
    
    Args:
        start: [x, y] start point
        end: [x, y] end point
        n_segments: number of segments
        curvature: amount of curvature (0 = straight, 0.5 = very curved)
    
    Returns:
        points: (n+1, 2) array of chain points
    """
    n_points = n_segments + 1
    points = np.zeros((n_points, 2))
    
    # Create evenly spaced points along backbone
    for i in range(n_points):
        t = i / (n_segments)
        points[i] = start + t * (end - start)
    
    # Add perpendicular curvature (sine wave)
    chain_vec = end - start
    chain_length = np.linalg.norm(chain_vec)
    perp_dir = np.array([-chain_vec[1], chain_vec[0]]) / chain_length
    
    for i in range(1, n_points - 1):
        t = i / n_segments
        # Sine wave for smooth curvature
        offset = curvature * chain_length * np.sin(np.pi * t)
        points[i] += offset * perp_dir
    
    return points


def rotate_offsets_complex(old_points, new_start, new_end):
    """
    Rotate perpendicular offsets using complex numbers.
    
    Args:
        old_points: (n, 2) array of old chain points
        new_start: [x, y] new start point
        new_end: [x, y] new end point
    
    Returns:
        new_points: (n, 2) array with rotated offsets
    """
    n_points = len(old_points)
    new_points = np.zeros_like(old_points)
    
    # Set endpoints
    new_points[0] = new_start
    new_points[-1] = new_end
    
    # Old chain configuration
    old_start = old_points[0]
    old_end = old_points[-1]
    old_chain_vec = old_end - old_start
    old_chain_complex = complex(old_chain_vec[0], old_chain_vec[1])
    
    # New chain configuration
    new_chain_vec = new_end - new_start
    new_chain_complex = complex(new_chain_vec[0], new_chain_vec[1])
    
    # Rotation (without scaling)
    rotation_complex = new_chain_complex / old_chain_complex
    rotation_complex = rotation_complex / abs(rotation_complex)
    
    # Recompute intermediate points with rotated offsets
    for i in range(1, n_points - 1):
        t = i / (n_points - 1)
        
        # Old offset from ideal position
        old_ideal = old_start + t * old_chain_vec
        old_offset_vec = old_points[i] - old_ideal
        old_offset_complex = complex(old_offset_vec[0], old_offset_vec[1])
        
        # Rotate offset
        new_offset_complex = old_offset_complex * rotation_complex
        
        # New ideal position
        new_ideal = new_start + t * new_chain_vec
        
        # Apply rotated offset
        new_points[i] = new_ideal + np.array([new_offset_complex.real, 
                                               new_offset_complex.imag])
    
    return new_points


def demonstrate_rotation():
    """Demonstrate the rotation behavior with visualization."""
    # Create initial curved chain
    start = np.array([100, 200])
    end = np.array([400, 200])
    n_segments = 5
    
    initial_chain = create_curved_chain(start, end, n_segments, curvature=0.3)
    
    # Rotate the chain by moving the endpoint
    # Rotate end point by 45 degrees around start
    angle = np.pi / 4  # 45 degrees
    center = start
    rel_pos = end - center
    new_end = center + np.array([
        rel_pos[0] * np.cos(angle) - rel_pos[1] * np.sin(angle),
        rel_pos[0] * np.sin(angle) + rel_pos[1] * np.cos(angle)
    ])
    
    # Compute new chain with rotated offsets
    rotated_chain = rotate_offsets_complex(initial_chain, start, new_end)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Initial chain
    ax1.set_aspect('equal')
    ax1.plot([start[0], end[0]], [start[1], end[1]], 'k--', 
             alpha=0.3, linewidth=1, label='Backbone')
    ax1.plot(initial_chain[:, 0], initial_chain[:, 1], 'b-o', 
             linewidth=2, markersize=8, label='Initial chain')
    ax1.plot(initial_chain[0, 0], initial_chain[0, 1], 'rs', markersize=12)
    ax1.plot(initial_chain[-1, 0], initial_chain[-1, 1], 'rs', markersize=12)
    ax1.set_title('Initial Chain', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(50, 450)
    ax1.set_ylim(100, 400)
    
    # Right plot: Rotated chain
    ax2.set_aspect('equal')
    ax2.plot([start[0], new_end[0]], [start[1], new_end[1]], 'k--', 
             alpha=0.3, linewidth=1, label='New backbone')
    ax2.plot(rotated_chain[:, 0], rotated_chain[:, 1], 'r-o', 
             linewidth=2, markersize=8, label='Rotated chain')
    ax2.plot(rotated_chain[0, 0], rotated_chain[0, 1], 'rs', markersize=12)
    ax2.plot(rotated_chain[-1, 0], rotated_chain[-1, 1], 'rs', markersize=12)
    
    # Draw initial chain in background for comparison
    ax2.plot(initial_chain[:, 0], initial_chain[:, 1], 'b-o', 
             linewidth=1, markersize=4, alpha=0.3, label='Initial (reference)')
    
    ax2.set_title('After Rotating Endpoint by 45°', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(50, 450)
    ax2.set_ylim(100, 400)
    
    # Add arrow showing rotation
    ax2.annotate('', xy=new_end, xytext=end,
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax2.text((end[0] + new_end[0])/2, (end[1] + new_end[1])/2 + 20, 
             '45° rotation', fontsize=10, color='green', fontweight='bold')
    
    plt.suptitle('Complex Number Rotation: Preserving Chain Curvature', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print angle information
    print("="*60)
    print("COMPLEX NUMBER CHAIN ROTATION DEMONSTRATION")
    print("="*60)
    print("\nInitial chain:")
    print(f"  Start: {start}")
    print(f"  End: {end}")
    print(f"  Backbone angle: {np.degrees(np.arctan2(end[1]-start[1], end[0]-start[0])):.1f}°")
    
    print("\nAfter rotation:")
    print(f"  New end: {new_end}")
    print(f"  New backbone angle: {np.degrees(np.arctan2(new_end[1]-start[1], new_end[0]-start[0])):.1f}°")
    
    print("\nKey insight:")
    print("  The SHAPE of the curvature is preserved!")
    print("  The chain maintains its 'bent' appearance,")
    print("  just rotated to match the new endpoint orientation.")
    print("="*60)


if __name__ == "__main__":
    print("This demonstrates how complex numbers preserve chain curvature")
    print("when endpoints are rotated.\n")
    
    demonstrate_rotation()
