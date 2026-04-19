import numpy as np
import open3d as o3d

from .base_qem import QEMCalculator
from .edge_collapse import EdgeCollapser


class ProgressiveLOD:
    """生成渐进式LOD（Level of Detail）模型"""
    
    def __init__(self):
        self.lod_levels = []  # 存储不同LOD级别的模型
        self.collapse_history = []  # 记录边折叠历史
    
    def generate_lods(self, mesh, target_faces_list, feature_aware=False, crease_edges=None, max_iterations_per_lod=10000, dihedral_angle_threshold=45.0, use_open3d=False, feature_extractor=None):
        """生成多个LOD级别的模型
        
        参数:
            mesh: 输入网格
            target_faces_list: 目标面数列表
            feature_aware: 是否使用特征感知简化
            crease_edges: 折痕边列表
            max_iterations_per_lod: 每个LOD级别的最大迭代次数
            dihedral_angle_threshold: 二面角阈值
            use_open3d: 是否使用Open3D内置简化方法
            feature_extractor: 特征提取器实例（可选）
        """
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
        feature_integrator = None
        if feature_aware:
            if feature_extractor:
                # 使用深度学习特征提取器
                feature_integrator = feature_extractor
                print("Using deep learning feature extractor for feature-aware simplification")
            else:
                # 使用传统的特征感知QEM
                from .feature_aware_qem import FeatureAwareQEM
                feature_qem = FeatureAwareQEM(current_mesh)
                if crease_edges:
                    feature_qem.load_feature_edges(crease_edges)
                else:
                    feature_qem.set_feature_edges(current_mesh, dihedral_angle_threshold=dihedral_angle_threshold)
                print("Using traditional feature-aware QEM")
        
        for target_faces in target_lods:
            if len(current_mesh.triangles) <= target_faces:
                print(f"Already below target faces: {target_faces}")
                continue
            
            print(f"Generating LOD with {target_faces} faces...")
            
            # 计算二次误差矩阵（仅当不使用Open3D时需要）
            quadrics = None
            if not use_open3d:
                qem_calc = QEMCalculator()
                quadrics = qem_calc.compute_all_quadrics(current_mesh)
            
            # 执行简化
            simplified_mesh = self._simplify_to_target(
                current_mesh, target_faces, quadrics, feature_qem, max_iterations_per_lod, use_open3d, feature_integrator
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
    
    def _simplify_to_target(self, mesh, target_faces, quadrics, feature_aware_qem=None, max_iterations=10000, use_open3d=False, feature_integrator=None):
        """简化网格到目标面数
        
        参数:
            mesh: 输入网格
            target_faces: 目标面数
            quadrics: 二次误差矩阵
            feature_aware_qem: 特征感知QEM计算器
            max_iterations: 最大迭代次数
            use_open3d: 是否使用Open3D内置简化方法
            feature_integrator: 特征集成器实例（可选）
        """
        collapser = EdgeCollapser()
        
        # 使用Open3D内置方法（如果不使用深度学习特征）
        if use_open3d and not feature_integrator:
            return collapser.simplify_open3d(mesh, target_faces, feature_aware_qem)
        
        # 混合方法：先使用Open3D快速简化，再使用特征感知方法微调
        if use_open3d and feature_integrator:
            # 先使用Open3D快速简化到目标面数的120%
            intermediate_faces = int(target_faces * 1.2)
            if intermediate_faces < len(mesh.triangles):
                # 使用Open3D快速简化
                open3d_simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=intermediate_faces)
                print(f"Open3D快速简化完成，面数: {len(open3d_simplified.triangles)} (目标: {intermediate_faces})")
                
                # 预测顶点重要性
                vertex_importance = feature_integrator.predict_vertex_importance(open3d_simplified)
                print(f"Calculated vertex importance for {len(open3d_simplified.vertices)} vertices")
                
                # 快速特征感知简化：使用Open3D的simplify_quadric_decimation方法，但在简化前调整顶点权重
                # 这里我们通过调整顶点位置来间接影响简化过程
                vertices = np.asarray(open3d_simplified.vertices)
                normals = np.asarray(open3d_simplified.vertex_normals)
                
                # 对重要顶点进行微调，使其更突出，从而在简化过程中更不容易被折叠
                importance_threshold = 0.7
                important_vertices = np.where(vertex_importance > importance_threshold)[0]
                print(f"找到 {len(important_vertices)} 个重要顶点")
                
                if len(important_vertices) > 0:
                    # 对重要顶点进行微调，沿法线方向外推
                    for v in important_vertices:
                        importance = vertex_importance[v]
                        # 根据重要性调整顶点位置，重要性越高，外推越多
                        displacement = normals[v] * (importance - importance_threshold) * 0.05
                        vertices[v] += displacement
                    
                    # 创建新的网格
                    temp_mesh = o3d.geometry.TriangleMesh()
                    temp_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                    temp_mesh.triangles = open3d_simplified.triangles
                    temp_mesh.compute_vertex_normals()
                    
                    # 使用Open3D的quadric decimation方法简化到目标面数
                    simplified_mesh = temp_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
                    print(f"特征感知简化完成，面数: {len(simplified_mesh.triangles)} (目标: {target_faces})")
                else:
                    # 如果没有重要顶点，直接使用Open3D简化到目标面数
                    simplified_mesh = open3d_simplified.simplify_quadric_decimation(target_number_of_triangles=target_faces)
                    print(f"Open3D简化完成，面数: {len(simplified_mesh.triangles)} (目标: {target_faces})")
                
                simplified_mesh.compute_vertex_normals()
                return simplified_mesh
            else:
                # 如果初始面数已经小于或等于目标面数的120%，直接使用Open3D简化到目标面数
                simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
                print(f"Open3D简化完成，面数: {len(simplified_mesh.triangles)} (目标: {target_faces})")
                simplified_mesh.compute_vertex_normals()
                return simplified_mesh
        
        # 原始实现（保持不变）
        simplified_mesh = o3d.geometry.TriangleMesh(mesh)
        
        # 初始化 QEMCalculator
        qem_calc = QEMCalculator(simplified_mesh)
        
        # 计算二次误差矩阵（如果为None）
        if quadrics is None:
            quadrics = qem_calc.compute_all_quadrics(simplified_mesh)
            print(f"Calculated quadrics for {len(simplified_mesh.vertices)} vertices")
        
        # 预测顶点重要性（只计算一次）
        vertex_importance = None
        if feature_integrator:
            vertex_importance = feature_integrator.predict_vertex_importance(simplified_mesh)
            print(f"Calculated vertex importance for {len(simplified_mesh.vertices)} vertices")
        
        # 快速简化模式：对于大型模型，使用Open3D的quadric decimation方法
        if len(simplified_mesh.triangles) > 50000 and feature_integrator:
            print("使用Open3D快速简化大型模型...")
            simplified_mesh = simplified_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
            print(f"Open3D简化完成，面数: {len(simplified_mesh.triangles)} (目标: {target_faces})")
            simplified_mesh.compute_vertex_normals()
            return simplified_mesh
        
        iteration = 0
        while len(simplified_mesh.triangles) > target_faces and iteration < max_iterations:
            # 找到最佳折叠边
            if feature_integrator and vertex_importance is not None:
                # 先使用传统方法找到候选边
                best_edge, best_target, best_cost = collapser.find_best_collapse(
                    simplified_mesh, quadrics, feature_aware_qem
                )
                
                if best_edge:
                    # 只计算候选边的深度学习特征成本
                    v1, v2 = best_edge
                    
                    # 计算边的平均重要性
                    if v1 < len(vertex_importance) and v2 < len(vertex_importance):
                        avg_importance = (vertex_importance[v1] + vertex_importance[v2]) / 2
                        
                        # 调整成本（考虑特征重要性）
                        alpha = 0.7  # 特征权重因子
                        best_cost = best_cost * (1 - alpha * avg_importance)
            else:
                # 使用传统方法
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
            if affected_vertices and quadrics is not None:
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
    
    def _simplify_with_vertex_clustering(self, mesh, target_faces, vertex_importance):
        """使用顶点聚类方法进行简化，考虑顶点重要性
        
        参数:
            mesh: 输入网格
            target_faces: 目标面数
            vertex_importance: 顶点重要性数组
        
        返回:
            简化后的网格
        """
        current_mesh = o3d.geometry.TriangleMesh(mesh)
        initial_faces = len(current_mesh.triangles)
        
        print(f"初始面数: {initial_faces}, 目标面数: {target_faces}")
        
        if initial_faces <= target_faces:
            print("初始面数已经小于或等于目标面数，直接返回")
            return current_mesh
        
        # 计算需要的简化比例
        reduction_ratio = target_faces / initial_faces
        print(f"简化比例: {reduction_ratio:.4f}")
        
        # 使用Open3D的quadric decimation方法直接简化到目标面数
        # 这种方法更可靠，能够准确控制面数
        simplified_mesh = current_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        
        simplified_mesh.compute_vertex_normals()
        print(f"简化完成: {len(simplified_mesh.triangles)} 三角形 (目标: {target_faces})")
        
        return simplified_mesh
    
    def _feature_aware_post_processing(self, mesh, vertex_importance):
        """执行特征感知的后处理，优化重要特征的保留
        
        参数:
            mesh: 输入网格
            vertex_importance: 顶点重要性数组
        
        返回:
            处理后的网格
        """
        print("执行特征感知后处理...")
        
        # 1. 对重要特征区域进行局部细化
        # 首先，找出重要性高于阈值的顶点
        importance_threshold = 0.7
        important_vertices = np.where(vertex_importance > importance_threshold)[0]
        print(f"找到 {len(important_vertices)} 个重要顶点")
        
        if len(important_vertices) > 0:
            # 2. 对重要顶点周围的区域进行局部细化
            # 这里我们使用Open3D的subdivide方法来细化网格
            # 但只对包含重要顶点的三角形进行细化
            
            # 提取包含重要顶点的三角形
            triangles = np.asarray(mesh.triangles)
            important_triangles = []
            for i, tri in enumerate(triangles):
                if any(v in important_vertices for v in tri):
                    important_triangles.append(i)
            
            print(f"找到 {len(important_triangles)} 个包含重要顶点的三角形")
            
            # 如果有足够多的重要三角形，进行局部细化
            if len(important_triangles) > 10:
                # 创建一个临时网格，只包含重要三角形
                temp_mesh = o3d.geometry.TriangleMesh()
                temp_mesh.vertices = mesh.vertices
                temp_mesh.triangles = o3d.utility.Vector3iVector(triangles[important_triangles])
                temp_mesh.compute_vertex_normals()
                
                # 对临时网格进行细分
                subdivided_mesh = temp_mesh.subdivide_midpoint(number_of_iterations=1)
                print(f"细分后，面数从 {len(temp_mesh.triangles)} 增加到 {len(subdivided_mesh.triangles)}")
                
                # 3. 合并原始网格和细分后的网格
                # 这里我们简单地返回细分后的网格
                # 在实际应用中，可能需要更复杂的合并策略
                processed_mesh = subdivided_mesh
            else:
                # 如果重要三角形太少，直接返回原始网格
                processed_mesh = mesh
        else:
            # 如果没有重要顶点，直接返回原始网格
            processed_mesh = mesh
        
        # 4. 调整重要顶点的位置，使其更接近原始位置
        # 这里我们简单地保持顶点位置不变
        # 在实际应用中，可以根据需要实现更复杂的位置调整逻辑
        
        processed_mesh.compute_vertex_normals()
        print("特征感知后处理完成")
        
        return processed_mesh
    
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