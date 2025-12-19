import numpy as np
import open3d as o3d
from tqdm import tqdm

class EdgeCollapser:
    """执行边折叠操作的工具类"""
    
    def __init__(self):
        self.vertex_map = {}
    
    def simplify(self, mesh, target_faces, quadrics, feature_weights=None, max_iterations=10000, verbose=True):
        """执行边折叠简化"""
        # 使用深拷贝创建网格副本
        simplified_mesh = o3d.geometry.TriangleMesh(mesh)
        
        # 如果已经低于目标面数，直接返回
        if len(simplified_mesh.triangles) <= target_faces:
            return simplified_mesh
        
        from .base_qem import QEMCalculator
        qem_calc = QEMCalculator(simplified_mesh)
        
        # 计算总迭代次数（初始面数 - 目标面数）
        initial_faces = len(simplified_mesh.triangles)
        total_iterations = initial_faces - target_faces
        
        iteration = 0
        
        # 使用tqdm创建进度条
        if verbose:
            pbar = tqdm(total=total_iterations, desc="网格简化进度", unit="面")
        
        while len(simplified_mesh.triangles) > target_faces and iteration < max_iterations:
            current_faces = len(simplified_mesh.triangles)
            
            # 找到最佳折叠边
            best_edge, best_target, best_cost = self.find_best_collapse(simplified_mesh, quadrics)
            
            if not best_edge:
                break
            
            # 执行边折叠
            v1, v2 = best_edge
            simplified_mesh, affected_vertices = self.collapse_edge(simplified_mesh, v1, v2, best_target, return_affected=True)
            
            # 更新进度条 - 基于实际减少的面数
            new_faces = len(simplified_mesh.triangles)
            faces_reduced = current_faces - new_faces
            if verbose and faces_reduced > 0:
                # 计算当前进度
                current_progress = pbar.n
                # 确保不会超过总迭代次数
                update_amount = min(faces_reduced, total_iterations - current_progress)
                if update_amount > 0:
                    pbar.update(update_amount)
            
            # 仅更新受影响顶点的二次误差矩阵，而不是所有顶点
            if affected_vertices:
                # 获取受影响的顶点列表（不重复）
                affected_vertices = list(set(affected_vertices))
                
                # 更新受影响顶点的二次误差矩阵
                vertices = np.asarray(simplified_mesh.vertices)
                triangles = np.asarray(simplified_mesh.triangles)
                
                # 对于每个受影响的顶点，重新计算其二次误差矩阵
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
                    quadrics[vertex_idx] = vertex_quadric
            
            iteration += 1
        
        # 关闭进度条
        if verbose:
            pbar.close()
        
        simplified_mesh.compute_vertex_normals()
        return simplified_mesh
    
    def get_edge_neighbors(self, v, edges):
        """获取与顶点v相连的所有顶点"""
        neighbors = set()
        for edge in edges:
            if v == edge[0]:
                neighbors.add(edge[1])
            elif v == edge[1]:
                neighbors.add(edge[0])
        return neighbors
    
    def build_edge_set(self, mesh):
        """从网格构建边集合"""
        triangles = np.asarray(mesh.triangles)
        
        # 为每个三角形生成三个边
        edges = np.concatenate((
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]]
        ))
        
        # 对每条边的两个顶点进行排序，确保一致性
        edges = np.sort(edges, axis=1)
        
        # 使用 numpy 的 unique 函数去除重复边，直接返回numpy数组
        unique_edges = np.unique(edges, axis=0)
        
        return unique_edges
    
    def collapse_edge(self, mesh, v1, v2, target_pos, return_affected=False):
        """执行边折叠操作，将v1折叠到target_pos"""
        vertices = np.asarray(mesh.vertices).copy()
        triangles = np.asarray(mesh.triangles).copy()
        
        # 收集受影响的顶点
        affected_vertices = set()
        
        # 1. 更新顶点坐标
        vertices[v1] = target_pos
        
        # 2. 构建顶点映射（将v2映射到v1）
        num_vertices = len(vertices)
        self.vertex_map = {i: i for i in range(num_vertices)}
        self.vertex_map[v2] = v1
        
        # 3. 更新三角形，将所有v2替换为v1
        new_triangles = []
        for tri in triangles:
            # 映射三角形顶点
            mapped_tri = [self.vertex_map[v] for v in tri]
            
            # 检查是否有重复顶点（会导致退化三角形）
            if len(set(mapped_tri)) == 3:
                # 重新排序顶点以确保一致性
                new_tri = tuple(sorted(mapped_tri))
                if new_tri not in new_triangles:
                    new_triangles.append(new_tri)
                    # 添加受影响的顶点
                    affected_vertices.update(new_tri)
        
        # 4. 移除被折叠的顶点v2
        # 创建新的顶点列表（不包含v2）
        new_vertices = []
        # 更新顶点映射，将所有大于v2的顶点索引减1
        final_vertex_map = {}
        new_index = 0
        
        for i in range(num_vertices):
            if i == v2:
                continue  # 跳过被折叠的顶点
            
            final_vertex_map[i] = new_index
            new_vertices.append(vertices[i])
            new_index += 1
        
        # 5. 最终更新三角形顶点索引
        final_triangles = []
        for tri in new_triangles:
            final_tri = tuple(sorted([final_vertex_map[v] for v in tri]))
            final_triangles.append(final_tri)
        
        # 6. 创建新的网格
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(np.array(new_vertices))
        new_mesh.triangles = o3d.utility.Vector3iVector(np.array(final_triangles))
        
        # 7. 清理重复的三角形
        new_mesh.remove_duplicated_triangles()
        new_mesh.remove_degenerate_triangles()
        
        if return_affected:
            # 映射受影响的顶点到新的索引
            mapped_affected = [final_vertex_map[v] for v in affected_vertices if v != v2]
            return new_mesh, mapped_affected
        
        return new_mesh
    
    def find_best_collapse(self, mesh, quadrics, feature_aware_qem=None):
        """找到最佳的边折叠候选"""
        vertices = np.asarray(mesh.vertices)
        edges = self.build_edge_set(mesh)  # 现在返回numpy数组
        
        best_cost = float('inf')
        best_edge = None
        best_target_pos = None
        
        # 直接使用edges数组，无需转换
        v1_indices = edges[:, 0]
        v2_indices = edges[:, 1]
        
        # 批量获取顶点位置
        v1_positions = vertices[v1_indices]
        v2_positions = vertices[v2_indices]
        mid_positions = (v1_positions + v2_positions) / 2
        
        # 批量计算每条边的Q矩阵
        Q_v1 = np.array([quadrics[i] for i in v1_indices])
        Q_v2 = np.array([quadrics[i] for i in v2_indices])
        Q_edges = Q_v1 + Q_v2
        
        # 定义向量化的误差计算函数
        def compute_errors(positions, Qs):
            # 为每个点添加1，形成齐次坐标
            ones = np.ones((positions.shape[0], 1))
            homogeneous = np.hstack((positions, ones))
            
            # 计算误差：v^T * Q * v
            errors = np.sum(homogeneous[:, None, :] @ Qs @ homogeneous[:, :, None], axis=(1, 2))
            return errors
        
        # 批量计算三种目标点的误差
        errors_v1 = compute_errors(v1_positions, Q_edges)
        errors_v2 = compute_errors(v2_positions, Q_edges)
        errors_mid = compute_errors(mid_positions, Q_edges)
        
        # 找到每条边的最小误差和对应的目标点
        min_errors = np.minimum(np.minimum(errors_v1, errors_v2), errors_mid)
        target_indices = np.argmin(np.stack([errors_v1, errors_v2, errors_mid]), axis=0)
        
        # 如果使用特征感知QEM，应用特征权重
        if feature_aware_qem:
            # 逐个处理边，应用特征权重
            for i in range(len(edges)):
                v1, v2 = edges[i]
                edge_tuple = (v1, v2)
                current_cost = min_errors[i]
                # 应用特征权重
                adjusted_cost = feature_aware_qem.compute_feature_edge_cost(edge_tuple, current_cost, vertices)
                min_errors[i] = adjusted_cost
        
        # 找到全局最佳边
        best_idx = np.argmin(min_errors)
        best_edge = tuple(edges[best_idx])
        best_cost = min_errors[best_idx]
        target_idx = target_indices[best_idx]
        
        # 根据目标索引确定目标位置
        v1, v2 = best_edge
        if target_idx == 0:
            best_target_pos = vertices[v1]
        elif target_idx == 1:
            best_target_pos = vertices[v2]
        else:
            best_target_pos = (vertices[v1] + vertices[v2]) / 2
        
        return best_edge, best_target_pos, best_cost
    
    def _compute_error(self, point, Q):
        """计算点的二次误差"""
        v = np.append(point, 1.0)
        return np.dot(v, np.dot(Q, v))

def iterative_simplification(mesh, target_faces, quadrics, feature_aware_qem=None, max_iterations=10000):
    """迭代执行边折叠简化"""
    collapser = EdgeCollapser()
    # 使用深拷贝创建网格副本
    simplified_mesh = o3d.geometry.TriangleMesh(mesh)
    
    iteration = 0
    while len(simplified_mesh.triangles) > target_faces and iteration < max_iterations:
        # 找到最佳折叠边
        best_edge, best_target, best_cost = collapser.find_best_collapse(
            simplified_mesh, quadrics, feature_aware_qem
        )
        
        if not best_edge:
            break
        
        # 执行边折叠
        v1, v2 = best_edge
        simplified_mesh = collapser.collapse_edge(simplified_mesh, v1, v2, best_target)
        
        # 更新二次误差矩阵
        from .base_qem import QEMCalculator
        qem_calc = QEMCalculator()
        quadrics = qem_calc.compute_all_quadrics(simplified_mesh)
        
        iteration += 1
        
        # 每100次迭代打印进度
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Faces: {len(simplified_mesh.triangles)}")
    
    simplified_mesh.compute_vertex_normals()
    return simplified_mesh