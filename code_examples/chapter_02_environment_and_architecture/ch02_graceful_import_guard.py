"""
Chapter 2: Python 3 Development Environment & LidarLens Architecture
Code Example 2.2: Graceful Degradation & Dynamic Import Guards Architecture.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第2章 2.1.3节
"""

from typing import Any, Callable
import functools
import lidarlens as ll


def require_dependency(package_name: str, install_hint: str):
    """Decorator factory enforcing graceful degradation when optional packages are missing."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                __import__(package_name)
            except ImportError:
                raise ImportError(
                    f"\n[LidarLens Import Guard Error]:\n"
                    f"Function '{func.__name__}' requires optional package '{package_name}'.\n"
                    f"Resolution: Please run '{install_hint}' in your terminal environment."
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Example guarded functions
@require_dependency("open3d", "pip install pylidarlens[vis] or pip install open3d")
def visualize_scene_in_3d(points):
    """Guarded 3D visualization function."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd


@require_dependency("torch", "pip install pylidarlens[torch] or pip install torch")
def run_neural_backbone_feature_extraction(points):
    """Guarded deep learning backbone function."""
    import torch
    tensor = torch.from_numpy(points)
    return tensor.mean(dim=0)


def main():
    print("=== Chapter 2: Graceful Degradation & Dynamic Import Guards ===")
    
    # 1. Test core functionality that never requires heavy dependencies
    buffer = ll.LidarBuffer(capacity=1000)
    print(f"Core LidarBuffer allocated successfully: capacity={buffer.capacity}")
    
    # 2. Test guarded functions
    # Testing torch guard (installed in our test env)
    try:
        import numpy as np
        pts = np.random.randn(10, 4).astype(np.float32)
        feat = run_neural_backbone_feature_extraction(pts)
        print(f"Guarded PyTorch function executed successfully: result shape={feat.shape}")
    except ImportError as e:
        print(f"Graceful degradation caught expected ImportError: {e}")
        
    # Testing open3d guard (not installed globally)
    try:
        import numpy as np
        pts = np.random.randn(10, 4).astype(np.float32)
        visualize_scene_in_3d(pts)
    except ImportError as e:
        print(f"\nSuccessfully demonstrated Open3D import guard:")
        print(str(e).splitlines()[1])
        print(str(e).splitlines()[2])
        print(str(e).splitlines()[3])

    print("\nChapter 2.2 verification completed successfully.\n")


if __name__ == "__main__":
    main()
