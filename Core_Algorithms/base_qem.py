import numpy as np


class QEMCalculator:
    """计算顶点二次误差矩阵"""
    
    def __init__(self, mesh=None):
        self.mesh = mesh
        self.quadrics = {}
        self.vertex_quadrics = {}
    
    def compute_vertex_quadric(self, vertex, triangle, vertices):
        """计算单个顶点的二次误差矩阵"""
        # 获取三角形的三个顶点
        v0 = vertices[triangle[0]]
        v1 = vertices[triangle[1]]
        v2 = vertices[triangle[2]]
        
        # 计算三角形法线
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        
        # 计算法线长度
        normal_length = np.linalg.norm(normal)
        if normal_length < 1e-8:
            return np.zeros((4, 4))
        
        # 单位法线
        normal = normal / normal_length
        
        # 计算平面方程 ax + by + cz + d = 0
        a, b, c = normal
        d = -np.dot(normal, v0)
        
        # 构建平面向量 [a, b, c, d]
        p = np.array([a, b, c, d])
        
        # 计算二次误差矩阵 K = p^T * p
        K = np.outer(p, p)
        
        return K
    
    def compute_all_quadrics(self, mesh=None):
        """计算所有顶点的二次误差矩阵"""
        if mesh is None and self.mesh is None:
            raise ValueError("Mesh must be provided")
        
        current_mesh = mesh if mesh is not None else self.mesh
        vertices = np.asarray(current_mesh.vertices)
        triangles = np.asarray(current_mesh.triangles)
        
        # 初始化所有顶点的二次误差矩阵为零矩阵
        num_vertices = len(vertices)
        self.quadrics = {i: np.zeros((4, 4)) for i in range(num_vertices)}
        
        # 对每个三角形，更新三个顶点的二次误差矩阵
        for triangle in triangles:
            for vertex_idx in triangle:
                K = self.compute_vertex_quadric(vertex_idx, triangle, vertices)
                self.quadrics[vertex_idx] += K
        
        return self.quadrics
    
    def compute_vertex_quadrics(self, mesh=None):
        """计算所有顶点的二次误差矩阵（与compute_all_quadrics功能相同，为兼容接口）"""
        self.vertex_quadrics = self.compute_all_quadrics(mesh)
        return self.vertex_quadrics
    
    def compute_edge_error(self, v1_idx, v2_idx, vertices):
        """计算边折叠的误差"""
        # 合并两个顶点的二次误差矩阵
        Q = self.quadrics[v1_idx] + self.quadrics[v2_idx]
        
        # 尝试将v1折叠到v2
        error1 = self._compute_point_error(v2_idx, vertices, Q)
        
        # 尝试将v2折叠到v1
        error2 = self._compute_point_error(v1_idx, vertices, Q)
        
        # 尝试将它们折叠到中点
        mid_point = (vertices[v1_idx] + vertices[v2_idx]) / 2
        error3 = self._compute_point_error_from_coords(mid_point, Q)
        
        # 返回最小误差和对应的目标点
        min_error = min(error1, error2, error3)
        
        if min_error == error1:
            return min_error, vertices[v2_idx]
        elif min_error == error2:
            return min_error, vertices[v1_idx]
        else:
            return min_error, mid_point
    
    def _compute_point_error(self, vertex_idx, vertices, Q):
        """计算给定点的二次误差"""
        v = vertices[vertex_idx]
        return self._compute_point_error_from_coords(v, Q)
    
    def _compute_point_error_from_coords(self, point, Q):
        """从坐标计算点的二次误差"""
        # 将点扩展为齐次坐标 [x, y, z, 1]
        v = np.append(point, 1.0)
        
        # 计算误差 e = v^T * Q * v
        error = np.dot(v, np.dot(Q, v))
        
        return error

def compute_edge_costs(mesh, quadrics):
    """计算所有边的折叠成本"""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # 构建边集合
    edges = set()
    for triangle in triangles:
        for i in range(3):
            v1, v2 = triangle[i], triangle[(i+1)%3]
            if v1 < v2:
                edges.add((v1, v2))
            else:
                edges.add((v2, v1))
    
    # 计算每条边的折叠成本
    edge_costs = []
    for v1, v2 in edges:
        # 合并二次误差矩阵
        Q = quadrics[v1] + quadrics[v2]
        
        # 尝试三个可能的目标点
        v1_pos = vertices[v1]
        v2_pos = vertices[v2]
        mid_pos = (v1_pos + v2_pos) / 2
        
        # 计算每个目标点的误差
        e1 = _compute_error(v1_pos, Q)
        e2 = _compute_error(v2_pos, Q)
        e3 = _compute_error(mid_pos, Q)
        
        # 选择最小误差
        min_error = min(e1, e2, e3)
        
        # 确定最佳目标点
        if min_error == e1:
            target_pos = v1_pos
        elif min_error == e2:
            target_pos = v2_pos
        else:
            target_pos = mid_pos
        
        edge_costs.append({
            'edge': (v1, v2),
            'cost': min_error,
            'target': target_pos
        })
    
    return edge_costs

def _compute_error(point, Q):
    """计算点的二次误差"""
    v = np.append(point, 1.0)
    return np.dot(v, np.dot(Q, v))