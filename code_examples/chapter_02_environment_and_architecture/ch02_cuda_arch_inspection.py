"""
Chapter 2: Python 3 Development Environment & LidarLens Architecture
Code Example 2.1: CUDA Hardware Architecture Introspection & Compilation Guard.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第2章 2.1.1节
"""

from typing import Dict, Any, Tuple


def inspect_cuda_environment() -> Dict[str, Any]:
    """Inspect local hardware acceleration environment and Compute Capability."""
    info: Dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "device_name": "N/A",
        "compute_capability": "N/A",
        "arch_flag": "N/A",
        "torch_version": "N/A"
    }
    
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_count"] = torch.cuda.device_count()
            device_id = torch.cuda.current_device()
            major, minor = torch.cuda.get_device_capability(device_id)
            info["device_name"] = torch.cuda.get_device_name(device_id)
            info["compute_capability"] = f"{major}.{minor}"
            info["arch_flag"] = f"sm_{major}{minor}"
            # Recommended TORCH_CUDA_ARCH_LIST export
            info["recommended_env"] = f'export TORCH_CUDA_ARCH_LIST="{major}.{minor}+PTX"'
        else:
            info["recommended_env"] = "# CPU fallback: CUDA not available on this system."
    except ImportError:
        info["recommended_env"] = "# PyTorch not installed. Install torch to enable GPU acceleration."
        
    return info


def main():
    print("=== Chapter 2: CUDA Hardware Architecture Introspection ===")
    env = inspect_cuda_environment()
    print(f"PyTorch Version: {env['torch_version']}")
    print(f"CUDA Available:  {env['cuda_available']}")
    print(f"Device Count:    {env['device_count']}")
    print(f"Device Name:     {env['device_name']}")
    print(f"Compute Cap:     {env['compute_capability']} ({env['arch_flag']})")
    print(f"Recommended Env: {env['recommended_env']}")
    print("Chapter 2.1 verification completed successfully.\n")


if __name__ == "__main__":
    main()
