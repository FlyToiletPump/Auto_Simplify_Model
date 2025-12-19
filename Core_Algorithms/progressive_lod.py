import numpy as np
import open3d as o3d

from .base_qem import QEMCalculator
from .edge_collapse import EdgeCollapser


class ProgressiveLOD:
    """生成渐进式LOD（Level of Detail）模型"""
    
    def __init__(self):
        self.lod_levels = []  # 存储不同LOD级别的模型
        self.collapse_history = []  # 记录边折叠历史
    
    def generate_lods(self, mesh, target_faces_list, feature_aware=False, crease_edges=None, max_iterations_per_lod=10000, dihedral_angle_threshold=45.0):
        """生成多个LOD级别的模型"""
        # 确保target_faces_list是有序的（从高到低）
        target_lods = target_faces_list.copy()
        target_lods.sort(reverse=True)
        
        current_mesh = o3d.geometry.TriangleMesh(mesh)
        self.lod_levels.append({
            'mesh': current_mesh,
            'faces': len(current_mesh.triangles),
            'vertices': len(current_mesh.vertices)
        })
        
        print(f"Initial mesh: {len(current_mesh.triangles)} faces")
        
        # 初始化特征感知QEM（如果需要）
        feature_qem = None
        if feature_aware:
            from .feature_aware_qem import FeatureAwareQEM
            feature_qem = FeatureAwareQEM(current_mesh)
            if crease_edges:
                feature_qem.load_feature_edges(crease_edges)
            else:
                feature_qem.set_feature_edges(current_mesh, dihedral_angle_threshold=dihedral_angle_threshold)
        
        for target_faces in target_lods:
            if len(current_mesh.triangles) <= target_faces:
                print(f"Already below target faces: {target_faces}")
                continue
            
            print(f"Generating LOD with {target_faces} faces...")
            
            # 计算二次误差矩阵
            qem_calc = QEMCalculator()
            quadrics = qem_calc.compute_all_quadrics(current_mesh)
            
            # 执行简化
            simplified_mesh = self._simplify_to_target(
                current_mesh, target_faces, quadrics, feature_qem, max_iterations_per_lod
            )
            
            # 保存当前LOD级别
            self.lod_levels.append({
                'mesh': simplified_mesh,
                'faces': len(simplified_mesh.triangles),
                'vertices': len(simplified_mesh.vertices)
            })
            
            current_mesh = simplified_mesh
        
        # 按面数从少到多排序LOD级别
        self.lod_levels.sort(key=lambda x: x['faces'])
        
        # 返回仅包含网格的列表以兼容调用者
        return [lod['mesh'] for lod in self.lod_levels]
    
    def _simplify_to_target(self, mesh, target_faces, quadrics, feature_aware_qem=None, max_iterations=10000):
        """简化网格到目标面数"""
        collapser = EdgeCollapser()
        simplified_mesh = o3d.geometry.TriangleMesh(mesh)
        
        # 初始化 QEMCalculator
        qem_calc = QEMCalculator(simplified_mesh)
        
        iteration = 0
        while len(simplified_mesh.triangles) > target_faces and iteration < max_iterations:
            # 找到最佳折叠边
            best_edge, best_target, best_cost = collapser.find_best_collapse(
                simplified_mesh, quadrics, feature_aware_qem
            )
            
            if not best_edge:
                break
            
            # 记录折叠历史
            self.collapse_history.append({
                'edge': best_edge,
                'target_pos': best_target,
                'iteration': iteration,
                'faces_before': len(simplified_mesh.triangles)
            })
            
            # 执行边折叠，获取受影响的顶点
            v1, v2 = best_edge
            simplified_mesh, affected_vertices = collapser.collapse_edge(
                simplified_mesh, v1, v2, best_target, return_affected=True
            )
            
            # 仅更新受影响顶点的二次误差矩阵
            if affected_vertices:
                # 转换为 numpy 数组以便高效计算
                vertices = np.asarray(simplified_mesh.vertices)
                triangles = np.asarray(simplified_mesh.triangles)
                
                # 初始化新的 quadrics 数组
                new_quadrics = {i: quadrics.get(i, np.zeros((4, 4))) for i in range(len(vertices))}
                
                # 对每个受影响的顶点，重新计算其二次误差矩阵
                for vertex_idx in affected_vertices:
                    # 查找与该顶点相邻的所有三角形
                    adjacent_triangles = []
                    for i, triangle in enumerate(triangles):
                        if vertex_idx in triangle:
                            adjacent_triangles.append(triangle)
                    
                    # 重新计算该顶点的二次误差矩阵
                    vertex_quadric = np.zeros((4, 4))
                    for triangle in adjacent_triangles:
                        K = qem_calc.compute_vertex_quadric(vertex_idx, triangle, vertices)
                        vertex_quadric += K
                    
                    # 更新二次误差矩阵
                    new_quadrics[vertex_idx] = vertex_quadric
                
                # 使用新的 quadrics 替代旧的
                quadrics = new_quadrics
            
            iteration += 1
        
        simplified_mesh.compute_vertex_normals()
        return simplified_mesh
    
    def get_lod(self, index):
        """获取指定索引的LOD模型"""
        if 0 <= index < len(self.lod_levels):
            return self.lod_levels[index]['mesh']
        return None
    
    def get_lod_by_faces(self, target_faces):
        """获取最接近目标面数的LOD模型"""
        if not self.lod_levels:
            return None
        
        # 找到面数最接近且不大于目标面数的LOD
        best_match = None
        min_diff = float('inf')
        
        for lod in self.lod_levels:
            if lod['faces'] <= target_faces:
                diff = target_faces - lod['faces']
                if diff < min_diff:
                    min_diff = diff
                    best_match = lod
        
        return best_match['mesh'] if best_match else self.lod_levels[0]['mesh']
    
    def smooth_transition(self, lod1, lod2, alpha):
        """在两个LOD级别之间进行平滑过渡"""
        # 注意：这是一个简化实现，实际的平滑过渡需要更复杂的几何插值
        # 这里简单地对顶点位置进行线性插值
        
        vertices1 = np.asarray(lod1.vertices)
        vertices2 = np.asarray(lod2.vertices)
        
        # 确保两个模型有相同的顶点数（仅用于演示）
        if len(vertices1) != len(vertices2):
            print("Warning: LODs have different vertex counts, cannot perform smooth transition")
            return lod2 if alpha > 0.5 else lod1
        
        # 顶点位置线性插值
        interpolated_vertices = (1 - alpha) * vertices1 + alpha * vertices2
        
        # 创建过渡模型
        transition_mesh = o3d.geometry.TriangleMesh()
        transition_mesh.vertices = o3d.utility.Vector3dVector(interpolated_vertices)
        transition_mesh.triangles = lod1.triangles  # 使用较高精度的三角形拓扑
        transition_mesh.compute_vertex_normals()
        
        return transition_mesh
    
    def save_lods(self, output_dir, base_name="lod_"):
        """保存所有LOD级别到文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for i, lod in enumerate(self.lod_levels):
            filename = f"{base_name}{i}_{lod['faces']}faces.ply"
            filepath = os.path.join(output_dir, filename)
            o3d.io.write_triangle_mesh(filepath, lod['mesh'], write_ascii=True)
            print(f"Saved LOD {i} to {filepath}")
    
    def load_lods(self, input_dir, pattern="lod_*_*faces.ply"):
        """从文件加载LOD级别"""
        import os
        import glob
        
        self.lod_levels = []
        
        # 找到所有匹配的文件
        filepaths = glob.glob(os.path.join(input_dir, pattern))
        
        for filepath in filepaths:
            # 提取面数信息
            filename = os.path.basename(filepath)
            try:
                # 解析文件名：lod_i_faces.ply
                parts = filename.split('_')
                faces = int(parts[-1].split('faces')[0])
            except (IndexError, ValueError):
                print(f"Could not parse face count from filename: {filename}")
                continue
            
            # 加载模型
            mesh = o3d.io.read_triangle_mesh(filepath)
            if len(mesh.triangles) > 0:
                self.lod_levels.append({
                    'mesh': mesh,
                    'faces': faces,
                    'vertices': len(mesh.vertices)
                })
        
        # 按面数排序
        self.lod_levels.sort(key=lambda x: x['faces'])
        
        print(f"Loaded {len(self.lod_levels)} LOD levels")
        return self.lod_levels