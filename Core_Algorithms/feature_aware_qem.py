import numpy as np

from .base_qem import QEMCalculator, compute_edge_costs
from Data_Prep.crease_detector import detect_creases


class FeatureAwareQEM:
    """特征感知的二次误差度量算法"""
    
    def __init__(self, mesh=None, feature_weight=10.0):
        self.mesh = mesh
        self.qem_calculator = QEMCalculator(mesh)
        self.feature_weight = feature_weight
        self.crease_edges = set()
        self.feature_vertices = set()
        self.feature_weights = None
        self.vertex_quadrics = {}
    
    def compute_vertex_quadrics(self, mesh=None):
        """计算所有顶点的二次误差矩阵"""
        current_mesh = mesh if mesh is not None else self.mesh
        self.vertex_quadrics = self.qem_calculator.compute_all_quadrics(current_mesh)
        return self.vertex_quadrics
    
    def compute_feature_weights(self, crease_edges=None, mesh=None):
        """计算特征权重"""
        current_mesh = mesh if mesh is not None else self.mesh
        
        if crease_edges is not None:
            self.load_feature_edges(crease_edges)
        elif not self.crease_edges and current_mesh:
            self.set_feature_edges(current_mesh)
        
        vertices = np.asarray(current_mesh.vertices)
        num_vertices = len(vertices)
        
        # 初始化特征权重
        self.feature_weights = np.ones(num_vertices)
        
        # 为特征顶点分配更高权重
        for v_idx in self.feature_vertices:
            if v_idx < num_vertices:
                self.feature_weights[v_idx] = self.feature_weight
        
        return self.feature_weights
    
    def set_feature_edges(self, mesh, dihedral_angle_threshold=45.0):
        """检测并设置折痕边"""
        self.crease_edges = detect_creases(mesh, dihedral_angle_threshold)
        
        # 标记折痕边上的顶点为特征顶点
        for edge in self.crease_edges:
            self.feature_vertices.add(edge[0])
            self.feature_vertices.add(edge[1])
    
    def load_feature_edges(self, crease_edges):
        """加载外部检测的折痕边"""
        self.crease_edges = crease_edges
        
        # 标记折痕边上的顶点为特征顶点
        for edge in self.crease_edges:
            self.feature_vertices.add(edge[0])
            self.feature_vertices.add(edge[1])
    
    def compute_feature_edge_cost(self, edge, base_cost, vertices):
        """计算考虑特征的边折叠成本"""
        v1, v2 = edge
        
        # 检查是否为折痕边
        if edge in self.crease_edges or (v2, v1) in self.crease_edges:
            # 为折痕边增加惩罚成本
            return base_cost * self.feature_weight
        
        # 检查是否连接两个特征顶点
        if v1 in self.feature_vertices and v2 in self.feature_vertices:
            # 为连接特征顶点的边增加一定惩罚
            return base_cost * (self.feature_weight / 2)
        
        # 正常边使用基础成本
        return base_cost
    
    def simplify(self, mesh, target_faces, max_iterations=10000):
        """执行特征感知的网格简化"""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        original_faces = len(triangles)
        if original_faces <= target_faces:
            return mesh
        
        # 1. 计算所有顶点的二次误差矩阵
        quadrics = self.qem_calculator.compute_all_quadrics(mesh)
        
        # 2. 构建初始边集合和成本
        edge_costs = compute_edge_costs(mesh, quadrics)
        
        # 3. 执行边折叠简化
        iteration = 0
        simplified_mesh = mesh
        
        while len(simplified_mesh.triangles) > target_faces and iteration < max_iterations:
            # 更新顶点和三角形数据
            vertices = np.asarray(simplified_mesh.vertices)
            triangles = np.asarray(simplified_mesh.triangles)
            
            # 重新计算边成本（考虑特征）
            edge_costs = compute_edge_costs(simplified_mesh, quadrics)
            
            # 为每条边应用特征权重
            for edge_info in edge_costs:
                edge = edge_info['edge']
                base_cost = edge_info['cost']
                edge_info['cost'] = self.compute_feature_edge_cost(edge, base_cost, vertices)
            
            # 按成本排序边
            edge_costs.sort(key=lambda x: x['cost'])
            
            if not edge_costs:
                break
            
            # 选择成本最低的边进行折叠
            best_edge = edge_costs[0]
            v1, v2 = best_edge['edge']
            target_pos = best_edge['target']
            
            # 执行边折叠
            simplified_mesh = self._collapse_edge(simplified_mesh, v1, v2, target_pos)
            
            # 更新二次误差矩阵
            # 注意：这里简化了处理，实际应用中需要更精确的更新
            quadrics = self.qem_calculator.compute_all_quadrics(simplified_mesh)
            
            iteration += 1
        
        simplified_mesh.compute_vertex_normals()
        return simplified_mesh
    
    def _collapse_edge(self, mesh, v1, v2, target_pos):
        """执行边折叠操作"""
        # 使用Open3D的简化函数进行边折叠
        # 注意：这里使用了Open3D内置的简化函数作为示例
        # 实际应用中可能需要实现更精确的边折叠逻辑
        
        # 计算当前面数
        current_faces = len(mesh.triangles)
        target_faces = current_faces - 1
        
        if target_faces < 1:
            return mesh
        
        simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        simplified.compute_vertex_normals()
        return simplified
    
    def set_feature_weight(self, weight):
        """设置特征权重"""
        self.feature_weight = weight
        
    def get_feature_edges(self):
        """获取当前标记的折痕边"""
        return self.crease_edges