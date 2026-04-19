#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实3D模型加载器
"""

import os
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset, DataLoader

class RealModelDataset(Dataset):
    """真实3D模型数据集"""
    
    def __init__(self, data_dir, transform=None):
        """
        参数:
            data_dir: 模型数据目录
            transform: 数据变换函数
        """
        self.data_dir = data_dir
        self.transform = transform
        self.model_paths = self._collect_model_paths()
        
    def _collect_model_paths(self):
        """收集所有模型文件路径"""
        model_paths = []
        
        # 遍历所有子目录
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file == "model_normalized.obj":
                    model_paths.append(os.path.join(root, file))
        
        print(f"Found {len(model_paths)} model files")
        return model_paths
    
    def __len__(self):
        return len(self.model_paths)
    
    def __getitem__(self, idx):
        """获取一个样本"""
        model_path = self.model_paths[idx]
        
        # 加载模型
        mesh = o3d.io.read_triangle_mesh(model_path)
        
        # 计算顶点法线
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        # 提取顶点特征
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        
        # 合并坐标和法线作为特征
        features = np.concatenate([vertices, normals], axis=1)
        
        # 生成标签（基于几何属性）
        labels = self._generate_labels(mesh, vertices)
        
        # 应用变换
        if self.transform:
            features, labels = self.transform(features, labels)
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
    
    def _generate_labels(self, mesh, vertices):
        """生成标签"""
        # 计算顶点的度（连接的边数）
        triangles = np.asarray(mesh.triangles)
        vertex_degree = np.zeros(len(vertices), dtype=np.float32)
        
        for tri in triangles:
            vertex_degree[tri[0]] += 1
            vertex_degree[tri[1]] += 1
            vertex_degree[tri[2]] += 1
        
        # 计算顶点的曲率
        vertex_curvature = self._compute_vertex_curvature(mesh)
        
        # 计算顶点的边界性
        is_boundary = self._compute_boundary_vertices(mesh)
        
        # 合并标签
        # 重要性分数 = 度权重 + 曲率权重 + 边界权重
        importance = 0.3 * vertex_degree / (vertex_degree.max() + 1e-8) + \
                    0.5 * vertex_curvature + \
                    0.2 * is_boundary
        
        # 归一化到[0, 1]
        if importance.max() > importance.min():
            importance = (importance - importance.min()) / (importance.max() - importance.min())
        
        return importance
    
    def _compute_vertex_curvature(self, mesh):
        """计算顶点曲率"""
        # 简单的曲率计算：基于相邻顶点的法向量差异
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        triangles = np.asarray(mesh.triangles)
        
        curvature = np.zeros(len(vertices), dtype=np.float32)
        
        # 构建顶点邻居
        neighbors = {i: set() for i in range(len(vertices))}
        for tri in triangles:
            neighbors[tri[0]].add(tri[1])
            neighbors[tri[0]].add(tri[2])
            neighbors[tri[1]].add(tri[0])
            neighbors[tri[1]].add(tri[2])
            neighbors[tri[2]].add(tri[0])
            neighbors[tri[2]].add(tri[1])
        
        # 计算每个顶点的曲率
        for i in range(len(vertices)):
            if len(neighbors[i]) == 0:
                continue
            
            # 计算与邻居法线的平均差异
            normal_diff = 0.0
            for neighbor in neighbors[i]:
                # 计算法线夹角的正弦值（表示法线差异）
                dot_product = np.dot(normals[i], normals[neighbor])
                # 确保dot_product在[-1, 1]范围内
                dot_product = np.clip(dot_product, -1.0, 1.0)
                normal_diff += np.sqrt(1 - dot_product**2)
            
            curvature[i] = normal_diff / len(neighbors[i])
        
        # 归一化
        if curvature.max() > curvature.min():
            curvature = (curvature - curvature.min()) / (curvature.max() - curvature.min())
        
        return curvature
    
    def _compute_boundary_vertices(self, mesh):
        """计算边界顶点"""
        triangles = np.asarray(mesh.triangles)
        
        # 构建边集合
        edges = set()
        for tri in triangles:
            # 按顺序存储边，确保一致性
            edge1 = tuple(sorted((tri[0], tri[1])))
            edge2 = tuple(sorted((tri[1], tri[2])))
            edge3 = tuple(sorted((tri[2], tri[0])))
            
            edges.add(edge1)
            edges.add(edge2)
            edges.add(edge3)
        
        # 构建边计数
        edge_count = {}
        for tri in triangles:
            edge1 = tuple(sorted((tri[0], tri[1])))
            edge2 = tuple(sorted((tri[1], tri[2])))
            edge3 = tuple(sorted((tri[2], tri[0])))
            
            edge_count[edge1] = edge_count.get(edge1, 0) + 1
            edge_count[edge2] = edge_count.get(edge2, 0) + 1
            edge_count[edge3] = edge_count.get(edge3, 0) + 1
        
        # 边界边是只出现一次的边
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        # 边界顶点是边界边上的顶点
        boundary_vertices = set()
        for edge in boundary_edges:
            boundary_vertices.add(edge[0])
            boundary_vertices.add(edge[1])
        
        # 生成边界顶点掩码
        is_boundary = np.zeros(len(np.asarray(mesh.vertices)), dtype=np.float32)
        for vertex in boundary_vertices:
            is_boundary[vertex] = 1.0
        
        return is_boundary

def get_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=4):
    """获取数据加载器"""
    dataset = RealModelDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return dataloader

def collate_fn(batch):
    """自定义collate函数，处理不同大小的网格"""
    features_list = []
    labels_list = []
    
    for features, labels in batch:
        features_list.append(features)
        labels_list.append(labels)
    
    return features_list, labels_list

if __name__ == "__main__":
    # 测试数据加载器
    data_dir = "e:\GP\Auto_Simplify_Model-main\Auto_Simplify_Model\model3d"
    dataloader = get_dataloader(data_dir, batch_size=2, num_workers=0)
    
    for i, (features_list, labels_list) in enumerate(dataloader):
        print(f"Batch {i+1}")
        for j, (features, labels) in enumerate(zip(features_list, labels_list)):
            print(f"  Model {j+1}: {features.shape[0]} vertices")
        if i >= 2:
            break
