# LidarLens (`pylidarlens`)

[![PyPI - Version](https://img.shields.io/pypi/v/lidarlens.svg)](https://pypi.org/project/lidarlens/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blueviolet.svg)](https://en.cppreference.com/w/cpp/20)
[![ROS 2](https://img.shields.io/badge/ROS%202-Humble%20%7C%20Iron-orange.svg)](https://docs.ros.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**LidarLens** (`lidarlens` / `pylidarlens`) is a high-performance, production-grade 3D LiDAR point cloud perception, odometry, mapping, and Agentic AI framework designed for Autonomous Driving, Mobile Robotics, and Embodied Artificial Intelligence.

It serves as the native companion software framework and reference implementation for the textbook:
> **《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》** (人民邮电出版社).

---

## 🌟 Key Architectural Pillars

```
+─────────────────────────────────────────────────────────────────────────────+
|                         LIDARLENS DUAL-SPEED ARCHITECTURE                   |
+─────────────────────────────────────────────────────────────────────────────+
|                                                                             |
|  [HIGH-FREQUENCY INNER LOOP (>=30 Hz / <33.3 ms)]                           |
|  Raw LiDAR Ingestion (Zero-Copy Buffer Protocol C++20 / NumPy)             |
|       │                                                                     |
|       ▼                                                                     |
|  TemporalPipeline (Sliding-Window RingBuffer + Dynamic Temporal SOR)        |
|       │                                                                     |
|       ├─► 2.5D Elevation Grid & Traversability Analysis (Sobel + Step Height)
|       ├─► Ground Segmentation (Patchwork++ Concentric Zones / Planar RANSAC)|
|       ├─► 3D Obstacle Clustering (Distance-Adaptive DBSCAN + 2.5D PCA OBB)  |
|       ├─► Scan-to-Submap GICP Odometry & Hessian Degeneracy Auditing        |
|       └─► 2D Occupancy Grid / 3D OctoMap with Temporal Exponential Decay   |
|                                                                             |
|  [COGNITIVE AGENTIC AI OUTER LOOP (Asynchronous / Event-Driven)]            |
|  5-Tuple 3D Scene Graph G = (V, E, A, T, P)                                |
|       │                                                                     |
|       ▼                                                                     |
|  Multimodal Spatial-Temporal Association (Mahalanobis Gating + Feature EWMA)|
|       │                                                                     |
|       ▼                                                                     |
|  Pydantic V2 Spatial Tool Contracts (QueryMapSchema, NavigateSchema)        |
|       │                                                                     |
|       ▼                                                                     |
|  RobotToolExecutor & Autonomous Agentic Spatial Decision Loop               |
+─────────────────────────────────────────────────────────────────────────────+
```

1. **Zero-Copy Memory Protocol (C++20 & NumPy Buffer Protocol):**
   - High-throughput C++20 memory sharing via `pybind11` Buffer Protocol (`pybind11/numpy.h`).
   - Exposes contiguous `(N, 4)` float32 arrays directly to NumPy with zero heap allocation or memory copying.
   - Provides a transparent, pure-vectorized Python fallback when C++ binaries are not compiled.

2. **Dual-Speed Pipeline & Latency Budgets:**
   - **Low-Latency Inner Loop (≥30 Hz / <33.3 ms):** Sliding-window `RingBuffer` processing, voxel downsampling, dynamic temporal statistical outlier removal (SOR), 2.5D elevation maps, Patchwork++ ground extraction, and scan-to-submap GICP odometry.
   - **Agentic AI Outer Loop (Asynchronous):** Pydantic-validated tool schemas, hierarchical 3D Scene Graphs $G=(V, E, A, T, P)$, multimodal spatial reasoning, and closed-loop robot tool execution.

3. **Deterministic & Production-Ready:**
   - Strict typing across Python 3.10+, NumPy/Google-style docstrings.
   - Global deterministic random seed locking across NumPy, PyTorch, and Python `random`.
   - Comprehensive graceful degradation for optional dependencies (`torch`, `open3d`, `cv2`, `rclpy`, `laspy`, `onnx`).

---

## 📚 Curriculum Chapter Mapping

| Chapter | Module / Namespace | Core Components & Capabilities |
| :--- | :--- | :--- |
| **Ch 1** | `lidarlens.core`, `lidarlens.io` | Zero-copy `PointFrame`, KITTI Velodyne .bin, PCD, and Autonomous Driving dataset parsing. |
| **Ch 2** | `lidarlens.core`, `lidarlens.benchmark` | Zero-copy `LidarBuffer`, `LidarFrame`, `LidarLensBenchmark` (AABB & throughput audit). |
| **Ch 3** | `lidarlens.io`, `lidarlens.pipeline`, `lidarlens.filters` | `Streamer`, `TemporalPipeline`, `RingBuffer`, `VoxelDownsample`, `DynamicTemporalSOR`. |
| **Ch 4** | `lidarlens.terrain`, `lidarlens.cluster`, `lidarlens.geometry` | Vectorized `create_elevation_map`, `analyze_traversability` (Sobel + step height), `PatchworkPlusPlus`, `RANSACGround`, `AdaptiveDBSCAN`, `get_pca_obb`. |
| **Ch 5** | `lidarlens.registration` | IMU SLERP `de_skew_pointcloud`, `LowLatencyGICP` (Scan-to-Submap), `detect_degeneracy` (Hessian block eigen-analysis). |
| **Ch 6** | `lidarlens.mapping`, `lidarlens.io.map_io` | `generate_standard_grid` (ROS 2 0/100/-1 format), `OctoMap3D` (log-odds & temporal decay $L_t = L_{t-1} e^{-\lambda \Delta t}$), `run_map_regression` audit. |
| **Ch 7** | `lidarlens.augment`, `lidarlens.augment.qa` | `AugmentationPipeline` (YAML config & seed locking), `polar_mix`, `mix3d`, `run_8point_quality_gates` (overlap, ground clearance, KL divergence). |
| **Ch 8** | `lidarlens.models`, `lidarlens.dataset.kitti`, `lidarlens.deploy` | `PointPillarsBackbone`, `PillarFeatureNet`, `ScatterBEV`, `check_kitti_subset`, `export_to_onnx` (Opset 14, dynamic axes). |
| **Ch 9** | `lidarlens.sim2real`, `lidarlens.filters` | Beer-Lambert attenuation $T(R) = \exp(-2\sigma R)$, Marshall-Palmer rain DSD, `rain_pointcloud_denoise` cascade, `GradientReversalFn` (DANN), `setup_tta_model` & `tta_step` (Tent). |
| **Ch 10** | `lidarlens.scenegraph`, `lidarlens.agent` | 5-tuple `SceneGraph3D` $G=(V, E, A, T, P)$, `associate_and_update` (Mahalanobis + Cosine similarity), Pydantic tool schemas, `RobotToolExecutor`, `agentic_spatial_loop`. |
| **Ch 11** | `lidarlens.ros2` | `LidarPerceptionNode` (`sensor_msgs/PointCloud2` forward ROI filter, latency profiling, and `/cmd_vel` reactive collision avoidance). |

---

## 🛠️ Directory Structure

```text
pylidarlens/
├── pyproject.toml / setup.py           # C++20 pybind11 build configuration
├── src/                                # High-performance C++ extensions
│   ├── bindings.cpp                    # pybind11 module exports
│   ├── buffer/buffer_protocol.cpp      # Zero-copy NumPy buffer & memory pool (Ch 2)
│   ├── index/kdtree.cpp                # Fast k-d Tree & Morton coder (Ch 3)
│   ├── filters/temporal_sor.cpp        # RingBuffer + dynamic temporal SOR (Ch 3, 9)
│   └── registration/gicp.cpp           # Voxelized GICP & Hessian extraction (Ch 5)
├── lidarlens/                          # Python package root
│   ├── __init__.py                     # Unified API exports: ll.Streamer, ll.TemporalPipeline, etc.
│   ├── core.py                         # C++ bindings wrapper, LidarBuffer & LidarFrame
│   ├── benchmark.py                    # LidarLensBenchmark (latency & throughput)
│   ├── cli.py                          # Click CLI commands (info, downsample, denoise, ground, dtm)
│   ├── io/                             # I/O: streamer.py, las_io.py, map_io.py
│   ├── pipeline.py                     # TemporalPipeline & sliding RingBuffer
│   ├── filters/                        # VoxelDownsample, DynamicTemporalSOR, PatchworkPlusPlus, RANSACGround
│   ├── terrain/                        # Vectorized elevation_map.py, traversability, gis_terrain.py
│   ├── geometry/                       # get_pca_obb, base_geometry.py
│   ├── cluster/                        # AdaptiveDBSCAN, Euclidean clustering
│   ├── registration/                   # deskew.py (IMU SLERP), gicp.py (LowLatencyGICP), degeneracy.py
│   ├── mapping/                        # generate_standard_grid (2D ROS 2), OctoMap3D (3D temporal decay)
│   ├── augment/                        # AugmentationPipeline, polar_mix, mix3d, qa.py (8-point gates)
│   ├── models/                         # PointPillarsBackbone, PillarFeatureNet, ScatterBEV
│   ├── sim2real/                       # noise_models.py (Beer-Lambert/rain), adaptation.py (DANN/Tent)
│   ├── scenegraph/                     # SceneGraph3D G=(V,E,A,T,P), associate_and_update
│   ├── agent/                          # Pydantic tool schemas, RobotToolExecutor, agentic_spatial_loop
│   ├── deploy/                         # onnx_export.py, trt_wrapper.py
│   ├── ros2/                           # LidarPerceptionNode (rclpy reactive avoidance)
│   └── vis/                            # compare_view, render_intensity_pointcloud
└── tests/                              # Automated pytest test suite
```

---

## 🚀 Installation

### 1. Prerequisites

- **Python:** 3.10, 3.11, or 3.12.
- **Operating System:** Linux (Ubuntu 20.04/22.04+), Windows 10/11 (MSVC), macOS (Apple Clang).
- **Optional C++20 Compiler:** Required only if compiling native C++ extensions:
  - Linux: `g++ >= 11` or `clang++ >= 13`
  - Windows: Visual Studio 2022 (MSVC v143+) with C++ Desktop Development workload
  - macOS: Xcode Command Line Tools (`clang++`)

### 2. Standard Installation

Install directly via `pip` from PyPI:

```bash
pip install lidarlens
```

Or clone the repository and install the core package in editable mode:

```bash
# Clone the repository
git clone https://github.com/ShinyDataTech/lidarlens.git
cd pylidarlens

# Install core dependencies (NumPy, SciPy, PyYAML, Pydantic)
pip install -e .
```

### 3. Modular Feature Extras

Install optional dependencies tailored to your engineering workflow:

```bash
# Computer Vision & Plotting (OpenCV, Matplotlib)
pip install -e ".[cv]"

# Deep Learning (PyTorch >= 2.0, Scikit-learn)
pip install -e ".[torch]"

# Agentic AI & Spatial Reasoning (Pydantic v2, NetworkX)
pip install -e ".[agent]"

# Edge Inference & Deployment (ONNX, ONNX Runtime)
pip install -e ".[onnx]"

# Open3D Point Cloud Visualization
pip install -e ".[open3d]"

# Geospatial LAS/LAZ Point Cloud I/O (Laspy, lazrs, PyProj)
pip install -e ".[las]"

# Command Line Interface (Click)
pip install -e ".[cli]"

# Full Development & Testing Suite
pip install -e ".[dev]"

# Install ALL capabilities in one command
pip install -e ".[all]"
```

### 4. Compiling C++20 Acceleration Extensions

`lidarlens` includes native C++20 extensions (`_lidarlens_cpp`) accelerating zero-copy memory buffers, k-d Trees, temporal SOR filtering, and Scan-to-Submap GICP odometry.

```bash
# Install pybind11 build dependency
pip install pybind11 setuptools wheel

# Compile extensions in-place
python setup.py build_ext --inplace
```

> [!NOTE]
> **Graceful Degradation:** If a C++20 compiler is not available on your system, compilation can be skipped safely. `lidarlens` automatically falls back to 100% vectorized NumPy routines without raising runtime import errors.

### 5. Verification (Smoke Test)

Verify that the installation was successful:

```bash
python -c "import lidarlens as ll; print(f'LidarLens v{ll.__version__} initialized successfully!')"
```

---

## 📖 Quick Start & Code Examples

### 1. Zero-Copy Buffer Protocol & Streaming (Chapters 2 & 3)

Manage point cloud memory with zero-copy buffer views and stream multi-format point clouds:

```python
import numpy as np
import lidarlens as ll

# 1. Allocate a zero-copy memory buffer
buffer = ll.LidarBuffer(capacity=100000)
raw_points = np.random.randn(50000, 4).astype(np.float32)
buffer.copy_from(raw_points)

# 2. Acquire zero-copy NumPy view (shares underlying C++ heap memory)
view = buffer.get_numpy_view()
print(f"Points in buffer: {view.shape}")

# 3. Stream frames from a directory (KITTI .bin / PCD / custom)
streamer = ll.Streamer(input_source="data/kitti/sequences/00/velodyne", file_format="kitti_bin")
for frame in streamer:
    print(f"Streamed Frame #{frame.frame_id}: {frame.num_points} points at {frame.timestamp:.2f}s")
    break
```

---

### 2. Real-Time Temporal Perception Pipeline (Chapter 3)

Run a sliding-window temporal pipeline with voxel downsampling and dynamic temporal statistical outlier removal:

```python
import numpy as np
import lidarlens as ll

# Configure sliding window pipeline (target >=30 Hz / <33.3 ms)
pipeline = ll.TemporalPipeline(window_size=5)
pipeline.add(ll.VoxelDownsample(leaf_size=0.1))
pipeline.add(ll.DynamicTemporalSOR(k_neighbors=20, base_std_ratio=1.5, distance_scale=0.02, temporal_votes=2))

# Process incoming LiDAR frames
raw_cloud = np.random.uniform(-30, 30, size=(40000, 4)).astype(np.float32)
frame = ll.LidarFrame(points=raw_cloud)
cleaned_frame = pipeline.process(frame)

print(f"Original: {len(frame)} -> Cleaned: {len(cleaned_frame)}")
print(f"Pipeline latency: {pipeline.get_profiling_stats()['pipeline_ms']:.2f} ms")
```

---

### 3. Vectorized Terrain Elevation & Traversability (Chapter 4)

Construct 2.5D elevation maps without nested loops and generate robot traversability costmaps:

```python
import numpy as np
from lidarlens.terrain import create_elevation_map, analyze_traversability

points = np.random.uniform(-20, 20, size=(50000, 4)).astype(np.float32)
points[:, 2] = 0.05 * points[:, 0]  # Gentle slope with z variations

# Vectorized 2.5D elevation grid (< 5ms)
z_map, z_max, z_min = create_elevation_map(
    points,
    grid_size=0.2,
    x_range=(-20.0, 20.0),
    y_range=(-20.0, 20.0)
)
print(f"Elevation grid shape: {z_map.shape}")

# Evaluate surface slope (Sobel) and step height discontinuities
costmap, traversable_mask = analyze_traversability(z_map, z_max, z_min, res=0.2)
print(f"Traversable cell ratio: {(traversable_mask == 255).mean():.2%}")
```

---

### 4. Ground Segmentation & 3D Obstacle Clustering (Chapter 4)

Extract ground surfaces using Patchwork++ Concentric Zone Model (CZM) and cluster non-ground obstacles:

```python
import numpy as np
from lidarlens.filters import PatchworkPlusPlus
from lidarlens.cluster import AdaptiveDBSCAN
from lidarlens.geometry import get_pca_obb

points = np.random.uniform(-15, 15, size=(10000, 4)).astype(np.float32)
points[:, 2] = 0.0  # Ground points

# Add synthetic non-ground obstacle cluster
obstacle = np.random.normal(loc=[5.0, 5.0, 1.0, 100.0], scale=[0.5, 0.5, 0.5, 5.0], size=(150, 4)).astype(np.float32)
all_points = np.vstack([points, obstacle])

# 1. Patchwork++ ground segmentation
patchwork = PatchworkPlusPlus(czm_rings=4, czm_sectors=16, max_range=30.0)
ground_mask, non_ground_mask = patchwork.segment(all_points)

# 2. Distance-adaptive DBSCAN clustering on non-ground points
non_ground_pts = all_points[non_ground_mask]
clusterer = AdaptiveDBSCAN(eps_min=0.3, alpha=0.015, min_samples=5)
labels = clusterer.fit_predict(non_ground_pts)

# 3. Fit 2.5D Oriented Bounding Box (OBB) via PCA for first cluster
cluster_0 = non_ground_pts[labels == 0]
if len(cluster_0) > 0:
    center, size, yaw = get_pca_obb(cluster_0)
    print(f"Cluster #0 OBB Center: {center}, Size: {size}, Heading: {yaw:.3f} rad")
```

---

### 5. Low-Latency GICP Odometry & Hessian Degeneracy (Chapter 5)

Perform Scan-to-Submap Generalized ICP and audit the 6x6 Gauss-Newton Hessian matrix for geometric degeneracy:

```python
import numpy as np
from lidarlens.registration import LowLatencyGICP, detect_degeneracy

# Initialize GICP frontend
gicp = LowLatencyGICP(leaf_size=0.2, max_iter=25)

target = np.random.uniform(-10, 10, size=(2000, 4)).astype(np.float32)
source = target + np.array([0.15, 0.05, 0.0, 0.0], dtype=np.float32)

gicp.set_target(target)
result = gicp.align(source)
print(f"Estimated Rigid Transform:\n{result['transform']}")

# Audit Hessian matrix in geometrically degenerate scenarios (e.g. tunnels or featureless corridors)
degeneracy = detect_degeneracy(result["hessian"], thresh_cond=100.0)
if degeneracy["is_degenerate"]:
    print(f"Warning: Registration degenerate along directions: {degeneracy['degenerate_directions']}")
else:
    print(f"Odometry well-constrained: Rotation Cond={degeneracy['condition_number_rotation']:.1f}, Trans Cond={degeneracy['condition_number_translation']:.1f}")
```

---

### 6. Standard ROS 2 Occupancy Grid & 3D OctoMap (Chapter 6)

Generate standard ROS 2 2D costmaps with safety dilation, and maintain a 3D OctoMap with exponential decay:

```python
import numpy as np
from lidarlens.mapping import generate_standard_grid, OctoMap3D

points = np.random.uniform(-10, 10, size=(5000, 3)).astype(np.float32)
points[:, 2] = np.random.uniform(0.2, 1.8, size=5000)

# 1. Standard ROS 2 2D Occupancy Grid with safety dilation
grid_2d, meta = generate_standard_grid(points, resolution=0.1, inflation_radius=0.3)
print(f"2D Occupancy Grid: {meta['width']}x{meta['height']} cells, lethal cells: {meta['inflated_cells']}")

# 2. Volumetric 3D OctoMap with ray-casting clearing and temporal exponential decay
octomap = OctoMap3D(resolution=0.2, decay_lambda=0.01)
octomap.update_scan(sensor_origin=np.zeros(3), hit_points=points[:200], delta_time=0.1, ray_cast=True)
octomap.apply_temporal_decay(delta_t=1.0)
print(f"Active 3D voxels in OctoMap: {len(octomap.voxels)}")
```

---

### 7. 3D Data Augmentation & 8-Point Quality Gates (Chapter 7)

Apply config-driven 3D point cloud augmentations with synchronized 3D bounding box coordinate transformations:

```python
import numpy as np
from lidarlens.augment import AugmentationPipeline, polar_mix, mix3d

# Define reproducible augmentation pipeline
pipeline = AugmentationPipeline({
    "transforms": [
        {"type": "GlobalRotation", "rotation_range": [-0.78, 0.78]},
        {"type": "GlobalScaling", "scale_range": [0.95, 1.05]},
        {"type": "RandomFlip3D", "p": 0.5}
    ],
    "seed": 42
})

sample = {
    "points": np.random.randn(2000, 4).astype(np.float32),
    "gt_boxes_3d": np.array([[2.0, 3.0, 0.0, 4.0, 1.8, 1.5, 0.2]], dtype=np.float32)
}

augmented_sample = pipeline(sample)
print(f"Augmented points: {augmented_sample['points'].shape}")
print(f"Transformed 3D box: {augmented_sample['gt_boxes_3d']}")
```

---

### 8. PointPillars 3D Perception & ONNX Deployment (Chapters 8 & 11)

Instantiate the lightweight PointPillars pseudo-image backbone and export it to an optimized ONNX computational graph:

```python
import torch
from lidarlens.models import PointPillarsBackbone
from lidarlens.deploy import export_to_onnx

# 1. Instantiate PointPillars 2D BEV Backbone
backbone = PointPillarsBackbone(in_channels=64, out_channels=256)

# 2. Dummy BEV pseudo-image tensor (Batch, Channels, Height, Width)
dummy_bev = torch.randn(1, 64, 200, 200)

# 3. Export to ONNX with dynamic batch dimension annotations
onnx_file = export_to_onnx(
    model=backbone,
    save_path="pointpillars_backbone.onnx",
    dummy_input=dummy_bev,
    opset_version=14
)
print(f"ONNX computational graph exported to: {onnx_file}")
```

---

### 9. Sim-to-Real Physical Weather Simulation (Chapter 9)

Simulate Beer-Lambert atmospheric beam attenuation and Marshall-Palmer raindrop scattering noise:

```python
import numpy as np
from lidarlens.sim2real import beer_lambert_attenuation, sample_marshall_palmer_rain

clean_points = np.random.uniform(-30, 30, size=(10000, 4)).astype(np.float32)
clean_points[:, 3] = np.random.uniform(50, 200, size=10000)  # Reflection intensities
ranges = np.linalg.norm(clean_points[:, :3], axis=1)

# 1. Beer-Lambert atmospheric attenuation in fog (sigma_ext = 0.02 m^-1)
attenuated_intensities = beer_lambert_attenuation(clean_points[:, 3], ranges, sigma_ext=0.02)
clean_points[:, 3] = attenuated_intensities

# 2. Marshall-Palmer raindrop backscattering false-alarm points
rain_noise = sample_marshall_palmer_rain(clean_points, rain_rate_mm_hr=25.0, seed=42)
corrupted_cloud = np.vstack([clean_points, rain_noise])
print(f"Original: {len(clean_points)} -> With Rain Noise: {len(corrupted_cloud)} points")
```

---

### 10. 3D Scene Graph & Agentic AI Spatial Loop (Chapter 10)

Construct a hierarchical 5-tuple 3D Scene Graph $G=(V, E, A, T, P)$ and execute a multi-turn spatial agentic loop:

```python
import numpy as np
from lidarlens.scenegraph import SceneGraph3D, SceneNode
from lidarlens.agent import (
    RobotToolExecutor,
    MockNavClient,
    MockSpatialLLMClient,
    agentic_spatial_loop
)

# 1. Build 3D Scene Graph environment
sg = SceneGraph3D()
sg.add_node(SceneNode(
    id="dock_01",
    label="charging_dock",
    pose=np.array([2.50, 1.20, 0.20], dtype=np.float32),
    layer=2,
    room="hallway"
))

# 2. Initialize physical robot tool executor
executor = RobotToolExecutor(scene_graph=sg, nav_client=MockNavClient())

# 3. Simulate embodied LLM planner
llm_client = MockSpatialLLMClient(target_obj="charging_dock", target_coords=(2.50, 1.20))

# 4. Run closed-loop spatial agentic reasoning
reply = agentic_spatial_loop(
    instruction="Locate the charging dock and navigate to it.",
    llm_client=llm_client,
    executor=executor,
    max_steps=3
)
print(f"Agent Response: {reply}")
```

---

### 11. ROS 2 Reactive Perception Node (Chapter 11)

Deploy a real-time ROS 2 node subscribing to `sensor_msgs/PointCloud2`, running sub-5ms ROI filtering, and publishing `/cmd_vel` collision avoidance:

```bash
# Direct ROS 2 CLI execution:
ros2 run lidarlens perception_node
```

```python
# Programmatic initialization:
from lidarlens.ros2 import LidarPerceptionNode

# Subscribes to /lidar/points, performs ROI forward collision checking,
# and issues safety stops to /cmd_vel with millisecond latency logging.
```

---

### 12. Command Line Interface (CLI)

Batch process point clouds directly from the command line:

```bash
# Inspect point cloud statistics and bounding box
lidarlens info raw_scan.laz

# Voxel downsampling (0.5m leaf size)
lidarlens downsample raw_scan.laz -v 0.5 -o downsampled.laz

# Statistical outlier removal (denoising)
lidarlens denoise downsampled.laz -m statistical --nb-neighbors 20 --std-ratio 2.0 -o clean.laz

# Planar ground extraction
lidarlens ground clean.laz -d 0.3 -o ground.laz

# Generate Digital Terrain Model (DTM) PNG
lidarlens dtm ground.laz -r 1.0 -o dtm.png

# Generate contour lines as GeoJSON
lidarlens contours ground.laz -i 5.0 -o contours.geojson

# Cloud-to-cloud distance comparison
lidarlens compare scan_a.laz scan_b.laz
```

---

### 13. Geospatial LAS/LAZ Legacy Workflows

Process airborne and terrestrial LAS/LAZ surveys with georeferenced metadata:

```python
import lidarlens as ll

# Load LAS/LAZ with full georeferenced headers and VLRs
pcd, header, vlrs, wkt = ll.read("airborne_survey.laz")
print(f"Loaded {header.point_count:,} points with CRS: {wkt[:40]}...")

# Preprocessing & ground separation
clean_pcd = ll.denoise(pcd, method="statistical", nb_neighbors=20, std_ratio=2.0)
ground_pcd, non_ground_pcd, plane = ll.extract_ground(clean_pcd)

# Rasterize Digital Terrain Model (DTM) and export colorized visualization
dtm, meta = ll.generate_dtm(ground_pcd, resolution=1.0, wkt=wkt)
ll.dem_to_png(dtm, meta, "dtm_visualization.png", colormap="terrain")
```

---

## 🧪 Running the Test Suite

`lidarlens` maintains comprehensive test coverage across both unit tests and textbook chapter examples.

### 1. Running Unit Tests with `pytest`

```bash
# Run all tests
pytest

# Run chapter-specific test modules
pytest tests/test_pipeline.py       # Core, Pipeline, Streamer & Buffer (Ch 2 & 3)
pytest tests/test_terrain.py        # 2.5D Elevation Maps & Traversability (Ch 4)
pytest tests/test_registration.py   # Motion Deskew, GICP & Degeneracy (Ch 5)
pytest tests/test_mapping.py        # Occupancy Grid & 3D OctoMap (Ch 6)
pytest tests/test_augment.py        # Data Augmentation & 8-point Gates (Ch 7)
pytest tests/test_models.py         # PointPillars Architecture & ONNX (Ch 8 & 11)
pytest tests/test_sim2real.py       # Weather Physics, DANN & Tent (Ch 9)
pytest tests/test_scenegraph.py     # 3D Scene Graphs G=(V,E,A,T,P) (Ch 10)
pytest tests/test_agent.py          # Pydantic Tools & Agentic Loop (Ch 10)
pytest tests/test_ros2.py           # ROS 2 Perception Node (Ch 11)
```

### 2. Running Textbook Master Code Examples Suite

Execute and audit 100% of all 33 standalone chapter examples:

```bash
python code_examples/check_all_examples.py
```

---

## ⚡ Performance Benchmarks

Benchmark memory throughput and processing latency:

```python
import lidarlens as ll

benchmark = ll.LidarLensBenchmark(num_points=1_000_000, seed=42)
report = benchmark.run_full_audit()

print(f"NumPy View Overhead:       {report['view_latency_us']:.2f} microseconds (Zero-Copy)")
print(f"AABB Calculation:          {report['aabb_latency_ms']:.2f} ms")
print(f"Memory Copy Bandwidth:     {report['copy_bandwidth_gbps']:.2f} GB/s")
```

---

## 📖 Citation

If you use **LidarLens** or the accompanying textbook in your academic research or engineering projects, please cite:

```bibtex
@book{liu2026lidarlens,
  title     = {Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战},
  author    = {Liu, Wei},
  publisher = {Posts & Telecom Press (人民邮电出版社)},
  year      = {2026},
  url       = {https://github.com/ShinyDataTech/lidarlens}
}
```

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for complete details.
