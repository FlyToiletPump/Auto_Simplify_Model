#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心算法模块测试
"""

import os
import sys
import unittest

import open3d as o3d

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入核心算法模块
from Core_Algorithms.base_qem import QEMCalculator
from Core_Algorithms.feature_aware_qem import FeatureAwareQEM
from Core_Algorithms.edge_collapse import EdgeCollapser
from Core_Algorithms.progressive_lod import ProgressiveLOD

# 导入数据预处理模块
from Data_Prep.mesh_cleaner import preprocess_mesh
from Data_Prep.crease_detector import detect_creases

class TestCoreAlgorithms(unittest.TestCase):
    """核心算法模块测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建一个测试网格
        self.test_mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        self.test_mesh = self.test_mesh.subdivide_midpoint(3)  # 细分以增加复杂度
        
        # 预处理网格
        self.preprocessed_mesh = preprocess_mesh(self.test_mesh)
        
        # 检测折痕边
        self.crease_edges = detect_creases(self.preprocessed_mesh, dihedral_angle_threshold=45)
    
    def test_qem_calculator(self):
        """测试QEM误差计算器"""
        print("测试QEM误差计算器...")
        
        # 创建QEM计算器并计算顶点二次误差
        qem_calculator = QEMCalculator(self.preprocessed_mesh)
        qem_calculator.compute_vertex_quadrics()
        
        # 验证计算结果
        self.assertEqual(len(qem_calculator.vertex_quadrics), len(self.preprocessed_mesh.vertices))
        
        # 检查二次误差矩阵的形状
        for quadric in qem_calculator.vertex_quadrics:
            self.assertEqual(quadric.shape, (4, 4))
        
        print("QEM误差计算器测试通过")
    
    def test_feature_aware_qem(self):
        """测试特征感知QEM"""
        print("测试特征感知QEM...")
        
        # 创建特征感知QEM计算器
        feature_qem = FeatureAwareQEM(self.preprocessed_mesh)
        feature_qem.compute_vertex_quadrics()
        feature_qem.compute_feature_weights(self.crease_edges)
        
        # 验证计算结果
        self.assertEqual(len(feature_qem.vertex_quadrics), len(self.preprocessed_mesh.vertices))
        self.assertEqual(len(feature_qem.feature_weights), len(self.preprocessed_mesh.vertices))
        
        # 检查特征权重是否合理
        for weight in feature_qem.feature_weights.values():
            self.assertGreaterEqual(weight, 1.0)  # 特征权重应该大于等于1
        
        print("特征感知QEM测试通过")
    
    def test_edge_collapser(self):
        """测试边折叠操作"""
        print("测试边折叠操作...")
        
        # 创建QEM计算器
        qem_calculator = QEMCalculator(self.preprocessed_mesh)
        qem_calculator.compute_vertex_quadrics()
        
        # 创建边折叠器并执行简化
        edge_collapser = EdgeCollapser(self.preprocessed_mesh)
        target_faces = max(10, len(self.preprocessed_mesh.triangles) // 2)
        
        simplified_mesh = edge_collapser.simplify(
            target_faces=target_faces,
            quadrics=qem_calculator.vertex_quadrics
        )
        
        # 验证简化结果
        self.assertLessEqual(len(simplified_mesh.triangles), target_faces * 1.1)  # 允许10%的误差
        self.assertGreater(len(simplified_mesh.triangles), 0)
        self.assertTrue(simplified_mesh.has_vertices())
        
        print("边折叠操作测试通过")
    
    def test_feature_aware_simplification(self):
        """测试特征感知简化"""
        print("测试特征感知简化...")
        
        # 创建特征感知QEM计算器
        feature_qem = FeatureAwareQEM(self.preprocessed_mesh)
        feature_qem.compute_vertex_quadrics()
        feature_qem.compute_feature_weights(self.crease_edges)
        
        # 执行特征感知简化
        edge_collapser = EdgeCollapser(self.preprocessed_mesh)
        target_faces = max(10, len(self.preprocessed_mesh.triangles) // 2)
        
        simplified_feature = edge_collapser.simplify(
            target_faces=target_faces,
            quadrics=feature_qem.vertex_quadrics,
            feature_weights=feature_qem.feature_weights
        )
        
        # 执行普通QEM简化作为对比
        qem_calculator = QEMCalculator(self.preprocessed_mesh)
        qem_calculator.compute_vertex_quadrics()
        
        simplified_basic = edge_collapser.simplify(
            target_faces=target_faces,
            quadrics=qem_calculator.vertex_quadrics
        )
        
        # 验证两种简化方法都能产生有效的结果
        self.assertLessEqual(len(simplified_feature.triangles), target_faces * 1.1)
        self.assertLessEqual(len(simplified_basic.triangles), target_faces * 1.1)
        
        print("特征感知简化测试通过")
    
    def test_progressive_lod(self):
        """测试渐进式LOD生成"""
        print("测试渐进式LOD生成...")
        
        # 创建LOD生成器
        lod_generator = ProgressiveLOD(self.preprocessed_mesh)
        
        # 生成LOD级别
        target_faces_list = [
            len(self.preprocessed_mesh.triangles) // 2,
            len(self.preprocessed_mesh.triangles) // 4,
            len(self.preprocessed_mesh.triangles) // 8
        ]
        
        lod_levels = lod_generator.generate_lods(
            target_faces_list=target_faces_list,
            feature_aware=True,
            crease_edges=self.crease_edges
        )
        
        # 验证LOD生成结果
        self.assertEqual(len(lod_levels), len(target_faces_list))
        
        # 检查LOD级别是否按顺序递减
        for i in range(len(lod_levels) - 1):
            self.assertGreater(
                len(lod_levels[i].triangles),
                len(lod_levels[i+1].triangles)
            )
        
        # 检查每个LOD级别是否达到目标面数
        for i, (lod_mesh, target_faces) in enumerate(zip(lod_levels, target_faces_list)):
            self.assertLessEqual(len(lod_mesh.triangles), target_faces * 1.1)
        
        print("渐进式LOD生成测试通过")
    
    def test_edge_cost_calculation(self):
        """测试边折叠成本计算"""
        print("测试边折叠成本计算...")
        
        # 创建QEM计算器
        qem_calculator = QEMCalculator(self.preprocessed_mesh)
        qem_calculator.compute_vertex_quadrics()
        
        # 创建边折叠器
        edge_collapser = EdgeCollapser(self.preprocessed_mesh)
        
        # 计算所有边的折叠成本
        edge_costs = edge_collapser.compute_edge_costs(qem_calculator.vertex_quadrics)
        
        # 验证成本计算结果
        self.assertGreater(len(edge_costs), 0)
        
        # 检查成本是否为非负数
        for edge, cost in edge_costs.items():
            self.assertGreaterEqual(cost, 0.0)
        
        print("边折叠成本计算测试通过")

if __name__ == "__main__":
    print("=== 核心算法模块测试 ===")
    unittest.main(verbosity=2)