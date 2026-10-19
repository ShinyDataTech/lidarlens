# 《Python三维空间感知与机器人视觉》配套代码示例 (Code Examples)

本目录包含《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第1章至第11章的所有经过生产级重构、严格类型标注与全自动化断言验证的配套实战代码示例。全部代码深度集成开源框架 **`pylidarlens`**（即 `import lidarlens as ll`），全面支持 C++20 扩展、零拷贝内存协议、PyTorch 深度学习以及具身智能体闭环。

---

## 目录结构与章节索引

```text
code_examples/
├── check_all_examples.py                               # 自动化一键全量测试套件 (100% 覆盖率验证)
├── README.md                                           # 示例索引与使用指南
│
├── chapter_01_intro/                                   # 第1章：三维空间感知与LiDAR全景图
│   └── ch01_pointframe_io.py                           # 零拷贝 PointFrame 读写与 KITTI/AV2 解析
│
├── chapter_02_environment_and_architecture/            # 第2章：工程环境与生产级架构
│   ├── ch02_cuda_arch_inspection.py                    # CUDA Compute Capability 硬件微架构审计
│   ├── ch02_graceful_import_guard.py                   # C++ / CUDA 动态链接与优雅降级 Import Guard
│   ├── ch02_benchmark_lidarlens.py                     # 百万级点云零拷贝视图、AABB 与吞吐量基准测试
│   └── ch02_render_intensity.py                        # Turbo 伪彩色映射与激光雷达反射率渲染
│
├── chapter_03_spatial_indexing_and_denoising/          # 第3章：空间索引与低延迟去噪
│   ├── ch03_spatial_indexing_kdtree_morton.py          # 3D Morton 空间填充曲线编码与 k-d Tree 近邻检索
│   └── ch03_streamer_temporal_pipeline.py              # 高通量流水线：体素降采样与时序动态 SOR 级联滤波
│
├── chapter_04_terrain_and_clustering/                  # 第4章：复杂地形与障碍物聚类
│   ├── ch04_elevation_and_traversability.py            # 向量化高程图栅格化与 Sobel 机器人通行度评估
│   ├── ch04_ground_segmentation_ransac_patchwork.py    # 平面 RANSAC 与 Patchwork++ 同心区域地面分割对比
│   └── ch04_adaptive_dbscan_and_pca_obb.py             # 距离自适应 DBSCAN、2.5D PCA 有向包围盒 (OBB) 与代价地图
│
├── chapter_05_registration_and_odometry/               # 第5章：点云配准与激光雷达里程计
│   ├── ch05_icp_mathematical_derivation.py             # SVD 交叉协方差 Point-to-Point ICP 数学推导与求解
│   ├── ch05_imu_slerp_deskew.py                        # IMU 高频位姿轨迹 SLERP 四元数插值运动去畸变
│   ├── ch05_low_latency_gicp_odometry.py               # 体素哈希地图与低延迟 GICP 连续帧里程计
│   └── ch05_degeneracy_detection.py                    # Hessian 块解耦、条件数与隧道/走廊退化方向辨识
│
├── chapter_06_occupancy_grid_mapping/                  # 第6章：栅格建图与导航感知融合
│   ├── ch06_standard_occupancy_grid.py                 # 标准 ROS 2 2D 占据栅格地图光栅化与膨胀层计算
│   ├── ch06_octomap_3d_temporal_decay.py               # 3D OctoMap 对数几率更新与时序指数衰减机制
│   └── ch06_nav2_map_regression_audit.py               # Nav2 PGM/YAML 离线地图连通域与孤岛噪点回归审计
│
├── chapter_07_data_augmentation_and_synthesis/         # 第7章：数据增强与合成质检门限
│   ├── ch07_augmentation_pipeline.py                   # MMDetection3D 风格同步 3D 边界盒刚体变换管线
│   ├── ch07_polarmix_and_mix3d.py                      # PolarMix 极坐标扇区交换与 Mix3D 场景多重叠加
│   └── ch07_qa_8point_quality_gates.py                 # 硬件级 8 项点云合成质检门限与防碰撞自动化审计
│
├── chapter_08_3d_object_detection_training/            # 第8章：3D目标检测与深度学习训练
│   ├── ch08_kitti_dataset_sanity_check.py              # KITTI 数据子集最小可行质量抽检 (Sanity Check)
│   ├── ch08_pointpillars_model_architecture.py         # PointPillars 动态柱化、PFN、Scatter 与 Backbone 全流程
│   └── ch08_amp_mixed_precision_training.py            # 自动混合精度 (AMP) 与 GradScaler 损失收敛训练循环
│
├── chapter_09_sim_to_real_adaptation/                  # 第9章：仿真到现实迁移与领域自适应
│   ├── ch09_beer_lambert_marshall_palmer.py            # 比尔-朗伯光衰减定律与马歇尔-帕尔默雨雾物理仿真
│   ├── ch09_rain_pointcloud_cascade_denoise.py         # 雨雾散射虚警点三级级联滤波 (强度门限 + ROR + SOR)
│   ├── ch09_dann_gradient_reversal.py                  # DANN 对抗性域自适应网络与梯度反转层 (GRL)
│   └── ch09_tent_test_time_adaptation.py               # 动态测试时自适应 (Tent) 最小化香农熵更新策略
│
├── chapter_10_scene_graph_and_agentic_ai/              # 第10章：3D场景图与具身智能体系统
│   ├── ch10_scenegraph_5tuple.py                       # 5元组分层 3D 场景图 G=(V,E,A,T,P) 拓扑构建与消歧查询
│   ├── ch10_multimodal_spatiotemporal_association.py   # 马氏距离门控与 CLIP/DINOv2 余弦特征 EWMA 动态融合
│   ├── ch10_pydantic_spatial_tools.py                  # 强类型 Pydantic 空间工具契约与 RobotToolExecutor 物理执行
│   └── ch10_agentic_spatial_loop.py                    # 具身双速智能体闭环：环境真实物理反馈注入与反重试风暴
│
└── chapter_11_edge_deployment_and_ros2/                # 第11章：边缘部署与ROS 2工程化
    ├── ch11_onnx_model_export.py                       # PointPillars Backbone 动态 Batch 维度 ONNX 计算图导出与校验
    └── ch11_ros2_lidar_perception_node.py              # ROS 2 实时点云感知、前向 ROI 避障与 /cmd_vel 控制闭环
```

---

## 运行与验证指南

### 1. 运行全局自动化测试套件
在项目根目录下运行一键测试脚本，将自动递归发现并运行所有章节的测试示例，打印单例耗时并断言结果：

```bash
python code_examples/check_all_examples.py
```

### 2. 单独运行特定章节示例
每个脚本均设计为 **完全自包含（Standalone）**，内置了确定性随机数种子、严谨的数学验证与单元断言。可以直接通过 Python 运行：

```bash
# 运行第 4 章：Patchwork++ 地面分割对比
python code_examples/chapter_04_terrain_and_clustering/ch04_ground_segmentation_ransac_patchwork.py

# 运行第 5 章：低延迟 GICP 连续帧里程计
python code_examples/chapter_05_registration_and_odometry/ch05_low_latency_gicp_odometry.py

# 运行第 8 章：PointPillars 架构与柱化前向推理
python code_examples/chapter_08_3d_object_detection_training/ch08_pointpillars_model_architecture.py

# 运行第 10 章：具身智能体闭环决策
python code_examples/chapter_10_scene_graph_and_agentic_ai/ch10_agentic_spatial_loop.py

# 运行第 11 章：ONNX 导出与验证
python code_examples/chapter_11_edge_deployment_and_ros2/ch11_onnx_model_export.py
```

---

## 核心技术特性
1. **零外部数据依赖**：所有需要数据集的章节示例（如 KITTI/AV2），均内置了高保真物理级合成数据生成器，即使没有下载数十 GB 的离线数据集，也能立即开箱运行并完成算法验证。
2. **多硬件环境自适应**：自动感知系统上的 CUDA/CPU 资源，在无 GPU 环境下自动无缝回退至 CPU 运算，同时保持 100% 的算法保真度。
3. **严格工业级质量**：所有代码严格遵循 PEP 8 规范、具有完整的类型注解（`typing`）、无任何未捕捉的异常或竞态条件。
