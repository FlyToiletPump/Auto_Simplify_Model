#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基本网格简化示例
展示如何使用我们的系统进行完整的网格简化流程
"""

import sys
import os
import open3d as o3d
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入数据预处理模块
from Data_Prep.data_loader import load_mesh, save_mesh
from Data_Prep.mesh_cleaner import preprocess_mesh
from Data_Prep.crease_detector import detect_creases, mark_creases

# 导入核心算法模块
from Core_Algorithms.base_qem import QEMCalculator
from Core_Algorithms.feature_aware_qem import FeatureAwareQEM
from Core_Algorithms.edge_collapse import EdgeCollapser
from Core_Algorithms.progressive_lod import ProgressiveLOD

# 导入评估模块
from Evaluation.metrics import MeshQualityEvaluator

def main(visualize=True, custom_mesh_path=None):
    """主函数
    
    Args:
        visualize: 是否可视化结果
        custom_mesh_path: 自定义模型路径
    """
    print("=== 网格简化系统示例 ===")
    
    # 1. 加载和预处理网格
    print("\n1. 加载和预处理网格")
    
    # 使用自定义模型路径或示例模型
    if custom_mesh_path and os.path.exists(custom_mesh_path):
        print(f"加载自定义模型: {custom_mesh_path}")
        mesh = load_mesh(custom_mesh_path)
    else:
        # 使用示例模型（如果存在）
        sample_mesh_path = "Auto_Simplify_Model/bunny_10k.obj"
        
        if os.path.exists(sample_mesh_path):
            print(f"加载示例模型: {sample_mesh_path}")
            mesh = load_mesh(sample_mesh_path)
        else:
            print("示例模型不存在，创建一个立方体用于演示")
            mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
            mesh = mesh.subdivide_midpoint(3)  # 细分以增加复杂度
    
    print(f"原始网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角形")
    
    # 预处理网格
    print("预处理网格...")
    preprocessed_mesh = preprocess_mesh(mesh)
    
    # 2. 检测折痕边
    print("\n2. 检测折痕边")
    crease_edges = detect_creases(preprocessed_mesh, dihedral_angle_threshold=45)
    print(f"检测到 {len(crease_edges)} 条折痕边")
    
    # 3. 基本QEM简化
    print("\n3. 执行基本QEM简化")
    target_faces_basic = max(100, len(preprocessed_mesh.triangles) // 10)  # 简化到10%的面数
    
    # 计算QEM误差
    qem_calculator = QEMCalculator(preprocessed_mesh)
    qem_calculator.compute_vertex_quadrics()
    
    # 执行边折叠
    edge_collapser = EdgeCollapser()
    simplified_basic = edge_collapser.simplify(
        mesh=preprocessed_mesh,
        target_faces=target_faces_basic,
        quadrics=qem_calculator.vertex_quadrics
    )
    
    print(f"基本QEM简化完成: {len(simplified_basic.triangles)} 三角形")
    
    # 4. 特征感知QEM简化
    print("\n4. 执行特征感知QEM简化")
    target_faces_feature = target_faces_basic
    
    feature_qem = FeatureAwareQEM(preprocessed_mesh)
    feature_qem.compute_vertex_quadrics()
    feature_qem.compute_feature_weights(crease_edges)
    
    simplified_feature = edge_collapser.simplify(
        mesh=preprocessed_mesh,
        target_faces=target_faces_feature,
        quadrics=feature_qem.vertex_quadrics,
        feature_weights=feature_qem.feature_weights
    )
    
    print(f"特征感知QEM简化完成: {len(simplified_feature.triangles)} 三角形")
    
    # 5. 渐进式LOD生成
    print("\n5. 生成渐进式LOD")
    lod_generator = ProgressiveLOD()
    lod_levels = lod_generator.generate_lods(
        mesh=preprocessed_mesh,
        target_faces_list=[5000, 2000, 1000, 500],
        feature_aware=True,
        crease_edges=crease_edges
    )
    
    print(f"生成了 {len(lod_levels)} 个LOD级别")
    for i, lod_mesh in enumerate(lod_levels):
        print(f"  LOD {i+1}: {len(lod_mesh.triangles)} 三角形")
    
    # 6. 质量评估
    print("\n6. 评估简化质量")
    evaluator = MeshQualityEvaluator(sample_points=5000)
    
    # 评估基本QEM
    metrics_basic = evaluator.evaluate_all(preprocessed_mesh, simplified_basic)
    print("\n基本QEM简化质量:")
    evaluator.print_metrics(metrics_basic)
    
    # 评估特征感知QEM
    metrics_feature = evaluator.evaluate_all(preprocessed_mesh, simplified_feature)
    print("\n特征感知QEM简化质量:")
    evaluator.print_metrics(metrics_feature)
    
    # 7. 保存结果
    print("\n7. 保存结果")
    save_mesh(simplified_basic, "../output/basic_simplified.ply")
    save_mesh(simplified_feature, "../output/feature_simplified.ply")
    
    # 保存LOD
    for i, lod_mesh in enumerate(lod_levels):
        save_mesh(lod_mesh, f"../output/lod_{i+1}.ply")
    
    # 8. 可视化（可选）
    if visualize:
        print("\n8. 可视化（按ESC退出）")
        visualize_meshes(preprocessed_mesh, simplified_basic, simplified_feature)
    else:
        print("\n8. 可视化已跳过")
    
    print("\n=== 示例完成 ===")

def visualize_meshes(original, simplified_basic, simplified_feature):
    """可视化原始网格和简化后的网格"""
    # 创建可视化窗口
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="网格简化结果对比", width=1200, height=800)
    
    # 添加原始网格
    original_copy = o3d.geometry.TriangleMesh(original)
    original_copy.paint_uniform_color([0.8, 0.8, 0.8])  # 灰色
    original_copy.translate([-2, 0, 0])  # 向左平移
    vis.add_geometry(original_copy)
    
    # 添加基本QEM简化后的网格
    basic_copy = o3d.geometry.TriangleMesh(simplified_basic)
    basic_copy.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
    vis.add_geometry(basic_copy)
    
    # 添加特征感知QEM简化后的网格
    feature_copy = o3d.geometry.TriangleMesh(simplified_feature)
    feature_copy.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
    feature_copy.translate([2, 0, 0])  # 向右平移
    vis.add_geometry(feature_copy)
    
    # 设置相机视角
    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_front([0, 0, -1])
    ctr.set_zoom(0.5)
    
    # 添加ESC退出回调
    vis.register_key_callback(256, lambda vis: vis.destroy_window())
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

import argparse

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="网格简化系统示例")
    parser.add_argument("--no-visualize", action="store_true", help="不显示可视化界面")
    parser.add_argument("--mesh-path", type=str, default=None, help="输入网格文件路径")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs("../output", exist_ok=True)
    
    # 运行主函数
    main(
        visualize=not args.no_visualize,
        custom_mesh_path=args.mesh_path
    )