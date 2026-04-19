#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实3D模型特征提取器
"""

import os
import torch
import numpy as np
import open3d as o3d

from Neural_Modules.feature_net import create_feature_net


class RealModelFeatureExtractor:
    """使用真实3D模型训练的特征提取器"""
    
    def __init__(self, model_path, device=None):
        """
        参数:
            model_path: 训练好的模型路径
            device: 运行设备 (None: 自动选择)
        """
        # 设备配置
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型
        self.model = create_feature_net(
            net_type="vertex",
            input_dim=6,  # 3D坐标 + 3D法线
            hidden_dim=64,
            output_dim=1
        )
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def extract_features(self, mesh):
        """
        从网格中提取特征
        
        参数:
            mesh: Open3D TriangleMesh对象
        
        返回:
            features: 顶点特征 [N, 6]
        """
        # 确保有法线
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        # 提取顶点和法线
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        
        # 合并特征
        features = np.concatenate([vertices, normals], axis=1)
        
        return features
    
    def compute_vertex_importance(self, mesh):
        """
        计算顶点重要性分数
        
        参数:
            mesh: Open3D TriangleMesh对象
        
        返回:
            importance: 顶点重要性分数 [N]
        """
        # 提取特征
        features = self.extract_features(mesh)
        
        # 转换为张量
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # 计算重要性分数
        with torch.no_grad():
            importance = self.model(features_tensor).squeeze().cpu().numpy()
        
        return importance
    
    def compute_edge_importance(self, mesh):
        """
        计算边重要性分数
        
        参数:
            mesh: Open3D TriangleMesh对象
        
        返回:
            edge_importance: 边重要性分数字典 {(v1, v2): score}
        """
        # 计算顶点重要性
        vertex_importance = self.compute_vertex_importance(mesh)
        
        # 构建边集合
        triangles = np.asarray(mesh.triangles)
        edges = set()
        
        for tri in triangles:
            edge1 = tuple(sorted((tri[0], tri[1])))
            edge2 = tuple(sorted((tri[1], tri[2])))
            edge3 = tuple(sorted((tri[2], tri[0])))
            
            edges.add(edge1)
            edges.add(edge2)
            edges.add(edge3)
        
        # 计算边重要性（取两个顶点的平均重要性）
        edge_importance = {}
        for edge in edges:
            v1, v2 = edge
            importance = (vertex_importance[v1] + vertex_importance[v2]) / 2
            edge_importance[edge] = importance
        
        return edge_importance
    
    def visualize_importance(self, mesh, importance):
        """
        可视化顶点重要性
        
        参数:
            mesh: Open3D TriangleMesh对象
            importance: 顶点重要性分数
        """
        # 归一化重要性到[0, 1]
        importance_normalized = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        
        # 创建颜色
        colors = np.zeros((len(importance), 3))
        
        # 红色表示重要，蓝色表示不重要
        for i in range(len(importance)):
            # 从蓝色(0,0,1)到红色(1,0,0)
            t = importance_normalized[i]
            colors[i] = [t, 0, 1 - t]
        
        # 设置顶点颜色
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        # 可视化
        o3d.visualization.draw_geometries([mesh], window_name="Vertex Importance Visualization")


def create_real_model_feature_extractor(model_path=None):
    """
    创建真实模型特征提取器实例
    
    参数:
        model_path: 模型路径，默认使用预训练模型
    
    返回:
        RealModelFeatureExtractor实例
    """
    if model_path is None:
        # 首先尝试使用真实模型训练的模型
        real_model_path = "Models\\real_model_feature_net.pth"
        if os.path.exists(real_model_path):
            model_path = real_model_path
        else:
            # 如果真实模型训练的模型不存在，使用合成数据训练的模型作为备选
            synth_model_path = "Models\\vertex_feature_net.pth"
            if os.path.exists(synth_model_path):
                model_path = synth_model_path
                print("Warning: Using synthetic data trained model as fallback")
            else:
                raise FileNotFoundError("No trained model found. Please run train_real_model_feature_net.py first.")
    
    return RealModelFeatureExtractor(model_path)
