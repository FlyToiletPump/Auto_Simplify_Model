#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理模块测试
"""

import os
import sys
import unittest

import open3d as o3d

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入数据预处理模块
from Data_Prep.data_loader import load_mesh, save_mesh
from Data_Prep.mesh_cleaner import clean_mesh, normalize_mesh, remove_noise, preprocess_mesh
from Data_Prep.crease_detector import detect_creases
from Data_Prep.noise_simulator import simulate_generative_output

class TestDataPrep(unittest.TestCase):
    """数据预处理模块测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建一个简单的测试网格
        self.test_mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        self.test_mesh = self.test_mesh.subdivide_midpoint(2)  # 细分以增加复杂度
        
        # 创建临时输出目录
        self.output_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后的清理工作"""
        # 删除临时文件
        import shutil
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
    
    def test_load_mesh(self):
        """测试加载网格功能"""
        print("测试加载网格功能...")
        
        # 保存测试网格
        test_file = os.path.join(self.output_dir, "test_mesh.obj")
        o3d.io.write_triangle_mesh(test_file, self.test_mesh)
        
        # 加载网格
        loaded_mesh = load_mesh(test_file)
        
        # 验证加载结果
        self.assertEqual(len(loaded_mesh.vertices), len(self.test_mesh.vertices))
        self.assertEqual(len(loaded_mesh.triangles), len(self.test_mesh.triangles))
        
        print("加载网格功能测试通过")
    
    def test_save_mesh(self):
        """测试保存网格功能"""
        print("测试保存网格功能...")
        
        # 保存测试网格
        test_file = os.path.join(self.output_dir, "test_save_mesh.ply")
        save_mesh(self.test_mesh, test_file)
        
        # 验证保存结果
        self.assertTrue(os.path.exists(test_file))
        
        # 重新加载并验证
        reloaded_mesh = o3d.io.read_triangle_mesh(test_file)
        self.assertEqual(len(reloaded_mesh.vertices), len(self.test_mesh.vertices))
        
        print("保存网格功能测试通过")
    
    def test_clean_mesh(self):
        """测试网格清理功能"""
        print("测试网格清理功能...")
        
        # 创建一个带有重复顶点的测试网格
        dirty_mesh = self.test_mesh.clone()
        dirty_mesh.vertices = o3d.utility.Vector3dVector(
            list(dirty_mesh.vertices) + list(dirty_mesh.vertices[:5])
        )
        
        # 清理网格
        cleaned_mesh = clean_mesh(dirty_mesh)
        
        # 验证清理结果
        self.assertLessEqual(len(cleaned_mesh.vertices), len(dirty_mesh.vertices))
        self.assertTrue(cleaned_mesh.is_watertight())
        
        print("网格清理功能测试通过")
    
    def test_normalize_mesh(self):
        """测试网格归一化功能"""
        print("测试网格归一化功能...")
        
        # 创建一个非归一化的测试网格
        non_normalized_mesh = self.test_mesh.clone()
        non_normalized_mesh.scale(2.0, center=non_normalized_mesh.get_center())
        non_normalized_mesh.translate([10.0, 10.0, 10.0])
        
        # 归一化网格
        normalized_mesh = normalize_mesh(non_normalized_mesh)
        
        # 验证归一化结果
        bounding_box = normalized_mesh.get_axis_aligned_bounding_box()
        max_dim = max(bounding_box.get_extent())
        self.assertAlmostEqual(max_dim, 2.0, delta=0.01)  # 应该缩放到直径2.0
        
        center = normalized_mesh.get_center()
        self.assertAlmostEqual(center[0], 0.0, delta=0.01)
        self.assertAlmostEqual(center[1], 0.0, delta=0.01)
        self.assertAlmostEqual(center[2], 0.0, delta=0.01)
        
        print("网格归一化功能测试通过")
    
    def test_remove_noise(self):
        """测试去噪功能"""
        print("测试去噪功能...")
        
        # 创建一个带有噪声的测试网格
        noisy_mesh = self.test_mesh.clone()
        vertices = np.asarray(noisy_mesh.vertices)
        noise = np.random.normal(0, 0.05, vertices.shape)
        noisy_mesh.vertices = o3d.utility.Vector3dVector(vertices + noise)
        
        # 去噪
        denoised_mesh = remove_noise(noisy_mesh)
        
        # 验证去噪结果
        noisy_vertices = np.asarray(noisy_mesh.vertices)
        denoised_vertices = np.asarray(denoised_mesh.vertices)
        
        # 计算噪声水平（顶点位移的标准差）
        noisy_std = np.std(noisy_vertices - np.asarray(self.test_mesh.vertices))
        denoised_std = np.std(denoised_vertices - np.asarray(self.test_mesh.vertices))
        
        self.assertLess(denoised_std, noisy_std)  # 去噪后的噪声应该更小
        
        print("去噪功能测试通过")
    
    def test_detect_creases(self):
        """测试折痕检测功能"""
        print("测试折痕检测功能...")
        
        # 创建一个有明显折痕的测试网格（立方体）
        cube_mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        
        # 检测折痕边
        crease_edges = detect_creases(cube_mesh, dihedral_angle_threshold=45)
        
        # 验证折痕检测结果
        # 立方体应该有12条边，都是折痕边
        self.assertEqual(len(crease_edges), 12)
        
        print("折痕检测功能测试通过")
    
    def test_preprocess_mesh(self):
        """测试完整的预处理流程"""
        print("测试完整的预处理流程...")
        
        # 创建一个带有噪声和异常的测试网格
        test_mesh = self.test_mesh.clone()
        
        # 添加噪声
        vertices = np.asarray(test_mesh.vertices)
        noise = np.random.normal(0, 0.05, vertices.shape)
        test_mesh.vertices = o3d.utility.Vector3dVector(vertices + noise)
        
        # 添加重复顶点
        test_mesh.vertices = o3d.utility.Vector3dVector(
            list(test_mesh.vertices) + list(test_mesh.vertices[:5])
        )
        
        # 缩放和偏移
        test_mesh.scale(2.0, center=test_mesh.get_center())
        test_mesh.translate([5.0, 5.0, 5.0])
        
        # 执行完整的预处理流程
        preprocessed_mesh = preprocess_mesh(test_mesh)
        
        # 验证预处理结果
        self.assertTrue(preprocessed_mesh.has_vertices())
        self.assertTrue(preprocessed_mesh.has_triangles())
        self.assertTrue(preprocessed_mesh.is_watertight())
        
        # 检查归一化
        bounding_box = preprocessed_mesh.get_axis_aligned_bounding_box()
        max_dim = max(bounding_box.get_extent())
        self.assertAlmostEqual(max_dim, 2.0, delta=0.01)
        
        print("完整预处理流程测试通过")

if __name__ == "__main__":
    # 如果需要导入numpy
    import numpy as np
    
    print("=== 数据预处理模块测试 ===")
    unittest.main(verbosity=2)