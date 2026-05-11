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

    def __init__(self, data_dir, transform=None, max_vertices=2000):
        """
        参数:
            data_dir: 模型数据目录
            transform: 数据变换函数
            max_vertices: 每个模型的最大顶点数
        """
        self.data_dir = data_dir
        self.transform = transform
        self.max_vertices = max_vertices
        self.model_paths = self._collect_model_paths()
        print(f"Total models collected: {len(self.model_paths)}")

    def _collect_model_paths(self):
        """收集所有模型文件路径"""
        model_paths = []

        # 遍历所有子目录
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                # 支持.obj和.stl文件
                if file == "model_normalized.obj" or file.endswith(".stl"):
                    model_paths.append(os.path.join(root, file))

        print(f"Found {len(model_paths)} model files")
        return model_paths

    def __len__(self):
        return len(self.model_paths)

    def __getitem__(self, idx):
        """获取一个样本"""
        model_path = self.model_paths[idx]

        try:
            # 加载模型
            mesh = o3d.io.read_triangle_mesh(model_path)

            # 检查模型是否有效
            if mesh.is_empty():
                raise ValueError("Mesh is empty")

            vertices = np.asarray(mesh.vertices)
            num_vertices = len(vertices)

            # 跳过顶点数为0的模型
            if num_vertices == 0:
                raise ValueError("Mesh has 0 vertices")

            # 跳过顶点数过大的模型（可能内存不足）
            if num_vertices > 5000000:
                raise ValueError(f"Mesh has too many vertices: {num_vertices}")

            # 计算顶点法线
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()

            normals = np.asarray(mesh.vertex_normals)

            # 限制顶点数
            if num_vertices > self.max_vertices:
                # 随机采样顶点
                sample_indices = np.random.choice(num_vertices, self.max_vertices, replace=False)
                sample_indices = np.sort(sample_indices)  # 排序以保持一致性
                vertices = vertices[sample_indices]
                normals = normals[sample_indices]
                num_vertices = self.max_vertices

            # 合并坐标和法线作为特征
            features = np.concatenate([vertices, normals], axis=1)

            # 生成标签（基于几何属性）
            labels = self._generate_labels(vertices)

            # 应用变换
            if self.transform:
                features, labels = self.transform(features, labels)

            return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

        except MemoryError as e:
            print(f"Memory error processing model {model_path}: {e}")
            return torch.empty(0, 6, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
        except Exception as e:
            # 如果处理失败，返回空数据
            print(f"Error processing model {model_path}: {e}")
            return torch.empty(0, 6, dtype=torch.float32), torch.empty(0, dtype=torch.float32)

    def _generate_labels(self, vertices):
        """生成标签（简化版本，减少计算量）"""
        num_vertices = len(vertices)

        # 简化标签：基于顶点在模型中的位置
        # 中心顶点更重要（通常在模型内部）
        center = vertices.mean(axis=0)
        distances = np.linalg.norm(vertices - center, axis=1)

        # 距离中心越远的顶点可能更重要（边界）
        # 距离中心越近的顶点可能更重要（特征点）
        importance = 1.0 - (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)

        return importance.astype(np.float32)


def get_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=0, max_vertices=2000):
    """获取数据加载器"""
    dataset = RealModelDataset(data_dir, max_vertices=max_vertices)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # Windows上建议设置为0
        collate_fn=collate_fn
    )
    return dataloader


def collate_fn(batch):
    """自定义collate函数，处理不同大小的网格"""
    features_list = []
    labels_list = []

    for features, labels in batch:
        # 过滤掉空数据
        if features.shape[0] > 0 and labels.shape[0] > 0:
            features_list.append(features)
            labels_list.append(labels)

    # 如果批次中没有有效数据，返回空列表
    if len(features_list) == 0:
        return [], []

    return features_list, labels_list


if __name__ == "__main__":
    # 测试数据加载器
    data_dir = "e:\\GP\\Auto_Simplify_Model-main\\Auto_Simplify_Model\\Thingi10K"
    dataloader = get_dataloader(data_dir, batch_size=2, num_workers=0, max_vertices=2000)

    for i, (features_list, labels_list) in enumerate(dataloader):
        print(f"Batch {i+1}")
        for j, (features, labels) in enumerate(zip(features_list, labels_list)):
            print(f"  Model {j+1}: {features.shape[0]} vertices")
        if i >= 2:
            break