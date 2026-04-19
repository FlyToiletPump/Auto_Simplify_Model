#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试真实3D模型特征提取模块
"""

import os
import open3d as o3d
import numpy as np

from Data_Prep.real_model_loader import get_dataloader
from Neural_Modules.real_model_feature_extractor import create_real_model_feature_extractor
from Core_Algorithms.progressive_lod import ProgressiveLOD


def test_real_model_feature_extractor():
    """测试真实模型特征提取器"""
    print("=== 测试真实模型特征提取器 ===")
    
    # 创建特征提取器
    extractor = create_real_model_feature_extractor()
    
    # 加载测试模型
    test_model_path = "e:\\GP\\Auto_Simplify_Model-main\\Auto_Simplify_Model\\bunny_10k.obj"
    if not os.path.exists(test_model_path):
        # 尝试从model3d目录获取一个模型
        model3d_dir = "e:\\GP\\Auto_Simplify_Model-main\\Auto_Simplify_Model\\model3d"
        model_paths = []
        for root, dirs, files in os.walk(model3d_dir):
            for file in files:
                if file == "model_normalized.obj":
                    model_paths.append(os.path.join(root, file))
        
        if model_paths:
            test_model_path = model_paths[0]
            print(f"Using test model: {test_model_path}")
        else:
            print("No test model found!")
            return
    
    # 加载模型
    mesh = o3d.io.read_triangle_mesh(test_model_path)
    mesh.compute_vertex_normals()
    
    print(f"Model loaded: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    
    # 提取特征
    features = extractor.extract_features(mesh)
    print(f"Features extracted: {features.shape}")
    
    # 计算顶点重要性
    vertex_importance = extractor.compute_vertex_importance(mesh)
    print(f"Vertex importance computed: {vertex_importance.shape}")
    print(f"Importance range: {vertex_importance.min():.4f} - {vertex_importance.max():.4f}")
    
    # 计算边重要性
    edge_importance = extractor.compute_edge_importance(mesh)
    print(f"Edge importance computed: {len(edge_importance)} edges")
    
    # 可视化重要性
    print("Visualizing vertex importance...")
    extractor.visualize_importance(mesh, vertex_importance)


def test_feature_integration():
    """测试特征集成到LOD生成中"""
    print("\n=== 测试特征集成到LOD生成中 ===")
    
    # 加载测试模型
    test_model_path = "e:\\GP\\Auto_Simplify_Model-main\\Auto_Simplify_Model\\bunny_10k.obj"
    if not os.path.exists(test_model_path):
        # 尝试从model3d目录获取一个模型
        model3d_dir = "e:\\GP\\Auto_Simplify_Model-main\\Auto_Simplify_Model\\model3d"
        model_paths = []
        for root, dirs, files in os.walk(model3d_dir):
            for file in files:
                if file == "model_normalized.obj":
                    model_paths.append(os.path.join(root, file))
        
        if model_paths:
            test_model_path = model_paths[0]
            print(f"Using test model: {test_model_path}")
        else:
            print("No test model found!")
            return
    
    # 加载模型
    mesh = o3d.io.read_triangle_mesh(test_model_path)
    mesh.compute_vertex_normals()
    
    print(f"Original model: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    
    # 创建特征提取器
    extractor = create_real_model_feature_extractor()
    
    # 创建LOD生成器
    lod_generator = ProgressiveLOD()
    
    # 生成LOD（使用Open3D内置方法，更快）
    target_faces = max(1000, len(mesh.triangles) // 10)
    print(f"Generating LOD with {target_faces} faces...")
    
    # 使用Open3D内置方法进行简化（更快）
    simplified_mesh = mesh.simplify_quadric_decimation(target_faces)
    simplified_mesh.compute_vertex_normals()
    
    print(f"Simplified model: {len(simplified_mesh.vertices)} vertices, {len(simplified_mesh.triangles)} faces")
    
    # 应用深度学习特征提取器到简化后的模型
    print("Applying deep learning feature extractor to simplified mesh...")
    vertex_importance = extractor.compute_vertex_importance(simplified_mesh)
    print(f"Vertex importance on simplified mesh: {vertex_importance.min():.4f} - {vertex_importance.max():.4f}")
    
    # 可视化结果
    print("Visualizing original and simplified models...")
    print("Note: Close the visualization window to continue...")
    o3d.visualization.draw_geometries([mesh], window_name="Original Model")
    o3d.visualization.draw_geometries([simplified_mesh], window_name="Simplified Model")


def test_data_loader():
    """测试真实模型数据加载器"""
    print("\n=== 测试真实模型数据加载器 ===")
    
    data_dir = "e:\\GP\\Auto_Simplify_Model-main\\Auto_Simplify_Model\\model3d"
    dataloader = get_dataloader(data_dir, batch_size=2, num_workers=0)
    
    print(f"Data loader created with {len(dataloader.dataset)} models")
    
    # 测试数据加载
    for i, (features_list, labels_list) in enumerate(dataloader):
        print(f"Batch {i+1}")
        for j, (features, labels) in enumerate(zip(features_list, labels_list)):
            print(f"  Model {j+1}: {features.shape[0]} vertices, features shape: {features.shape}")
            print(f"  Labels shape: {labels.shape}, label range: {labels.min().item():.4f} - {labels.max().item():.4f}")
        if i >= 1:
            break


if __name__ == "__main__":
    # 测试数据加载器
    test_data_loader()
    
    # 测试特征提取器
    test_real_model_feature_extractor()
    
    # 测试特征集成
    test_feature_integration()
