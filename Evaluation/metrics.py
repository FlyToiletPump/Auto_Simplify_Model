import os
import sys

import numpy as np
import open3d as o3d

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class MeshQualityEvaluator:
    """网格质量评估器"""
    
    def __init__(self, sample_points=10000):
        """
        初始化网格质量评估器
        
        参数:
            sample_points: 用于计算距离指标的采样点数量
        """
        self.sample_points = sample_points
    
    def compute_hausdorff_distance(self, mesh1, mesh2):
        """计算两个网格之间的Hausdorff距离"""
        # 从两个网格中采样点云
        pcd1 = mesh1.sample_points_uniformly(number_of_points=self.sample_points)
        pcd2 = mesh2.sample_points_uniformly(number_of_points=self.sample_points)
        
        # 计算双向Hausdorff距离
        dist1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
        dist2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
        
        hausdorff_dist = max(np.max(dist1), np.max(dist2))
        return hausdorff_dist
    
    def compute_rms_error(self, mesh1, mesh2):
        """计算两个网格之间的RMS误差"""
        pcd1 = mesh1.sample_points_uniformly(number_of_points=self.sample_points)
        pcd2 = mesh2.sample_points_uniformly(number_of_points=self.sample_points)
        
        # 计算点云距离
        dist = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
        rms = np.sqrt(np.mean(np.square(dist)))
        return rms
    
    def compute_chamfer_distance(self, mesh1, mesh2):
        """计算两个网格之间的Chamfer距离"""
        pcd1 = mesh1.sample_points_uniformly(number_of_points=self.sample_points)
        pcd2 = mesh2.sample_points_uniformly(number_of_points=self.sample_points)
        
        # 计算双向距离
        dist1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
        dist2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
        
        # Chamfer距离是双向平均距离
        chamfer_dist = (np.mean(dist1) + np.mean(dist2)) / 2
        return chamfer_dist
    
    def compute_normal_consistency(self, mesh1, mesh2):
        """计算法线一致性"""
        # 采样点云
        pcd1 = mesh1.sample_points_uniformly(number_of_points=self.sample_points)
        pcd2 = mesh2.sample_points_uniformly(number_of_points=self.sample_points)
        
        # 计算最近邻点
        pcd_tree = o3d.geometry.KDTreeFlann(pcd2)
        
        normal_consistency = []
        
        for i, point in enumerate(np.asarray(pcd1.points)):
            [_, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
            
            if idx:
                # 获取对应点的法线
                normal1 = np.asarray(pcd1.normals)[i]
                normal2 = np.asarray(pcd2.normals)[idx[0]]
                
                # 计算法线点积（一致性）
                dot_product = np.dot(normal1, normal2)
                dot_product = np.clip(dot_product, -1.0, 1.0)  # 避免浮点误差
                
                # 法线一致性是1减去角度的余弦值
                consistency = 1.0 - abs(dot_product)
                normal_consistency.append(consistency)
        
        if normal_consistency:
            return np.mean(normal_consistency)
        else:
            return 0.0
    
    def compute_crease_retention(self, original_mesh, simplified_mesh, dihedral_angle_threshold=45):
        """计算折痕边的保留率"""
        from Data_Prep.crease_detector import detect_creases
        
        # 检测两个网格中的折痕边
        orig_creases = detect_creases(original_mesh, dihedral_angle_threshold)
        simp_creases = detect_creases(simplified_mesh, dihedral_angle_threshold)
        
        if not orig_creases:
            return 1.0  # 如果原始网格没有折痕边，返回100%
        
        # 计算保留的折痕边数量
        retained = orig_creases.intersection(simp_creases)
        retention_rate = len(retained) / len(orig_creases)
        
        return retention_rate
    
    def compute_edge_collapse_ratio(self, original_mesh, simplified_mesh):
        """计算边折叠比率"""
        orig_edges = self._count_edges(original_mesh)
        simp_edges = self._count_edges(simplified_mesh)
        
        if orig_edges == 0:
            return 0.0
        
        return (orig_edges - simp_edges) / orig_edges
    
    def _count_edges(self, mesh):
        """计算网格中的边数"""
        triangles = np.asarray(mesh.triangles)
        edges = set()
        
        for tri in triangles:
            edges.add(tuple(sorted((tri[0], tri[1]))))
            edges.add(tuple(sorted((tri[1], tri[2]))))
            edges.add(tuple(sorted((tri[2], tri[0]))))
        
        return len(edges)
    
    def compute_vertex_reduction_ratio(self, original_mesh, simplified_mesh):
        """计算顶点减少比率"""
        orig_vertices = len(original_mesh.vertices)
        simp_vertices = len(simplified_mesh.vertices)
        
        if orig_vertices == 0:
            return 0.0
        
        return (orig_vertices - simp_vertices) / orig_vertices
    
    def compute_face_reduction_ratio(self, original_mesh, simplified_mesh):
        """计算面减少比率"""
        orig_faces = len(original_mesh.triangles)
        simp_faces = len(simplified_mesh.triangles)
        
        if orig_faces == 0:
            return 0.0
        
        return (orig_faces - simp_faces) / orig_faces
    
    def evaluate_all(self, original_mesh, simplified_mesh):
        """计算所有评估指标"""
        metrics = {
            'hausdorff_distance': self.compute_hausdorff_distance(original_mesh, simplified_mesh),
            'rms_error': self.compute_rms_error(original_mesh, simplified_mesh),
            'chamfer_distance': self.compute_chamfer_distance(original_mesh, simplified_mesh),
            'normal_consistency': self.compute_normal_consistency(original_mesh, simplified_mesh),
            'crease_retention': self.compute_crease_retention(original_mesh, simplified_mesh),
            'edge_collapse_ratio': self.compute_edge_collapse_ratio(original_mesh, simplified_mesh),
            'vertex_reduction_ratio': self.compute_vertex_reduction_ratio(original_mesh, simplified_mesh),
            'face_reduction_ratio': self.compute_face_reduction_ratio(original_mesh, simplified_mesh),
            'original_faces': len(original_mesh.triangles),
            'simplified_faces': len(simplified_mesh.triangles),
            'original_vertices': len(original_mesh.vertices),
            'simplified_vertices': len(simplified_mesh.vertices),
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """打印评估指标"""
        print("=== Mesh Simplification Quality Metrics ===")
        print(f"Original: {metrics['original_faces']} faces, {metrics['original_vertices']} vertices")
        print(f"Simplified: {metrics['simplified_faces']} faces, {metrics['simplified_vertices']} vertices")
        print(f"Face reduction: {metrics['face_reduction_ratio']:.2%}")
        print(f"Vertex reduction: {metrics['vertex_reduction_ratio']:.2%}")
        print("-")
        print(f"Hausdorff distance: {metrics['hausdorff_distance']:.6f}")
        print(f"RMS error: {metrics['rms_error']:.6f}")
        print(f"Chamfer distance: {metrics['chamfer_distance']:.6f}")
        print(f"Normal consistency: {metrics['normal_consistency']:.6f}")
        print(f"Crease retention: {metrics['crease_retention']:.2%}")
        print("==========================================")
    
    def save_metrics(self, metrics, output_path):
        """保存评估指标到JSON文件"""
        import json
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_path}")