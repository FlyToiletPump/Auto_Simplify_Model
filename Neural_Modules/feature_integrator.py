import numpy as np
import open3d as o3d
import torch

class FeatureIntegrator:
    """将深度学习特征与网格简化算法集成"""
    
    def __init__(self, feature_net=None):
        """
        参数:
            feature_net: 预训练的特征提取网络
        """
        self.feature_net = feature_net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def set_feature_net(self, feature_net):
        """设置特征提取网络"""
        self.feature_net = feature_net
        if feature_net:
            self.feature_net.to(self.device)
            self.feature_net.eval()
    
    def extract_vertex_features(self, mesh):
        """从网格中提取顶点特征"""
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        
        # 合并坐标和法线作为特征
        features = np.concatenate([vertices, normals], axis=1)
        
        return features
    
    def extract_edge_features(self, mesh):
        """从网格中提取边特征"""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # 构建边集合
        edges = set()
        for tri in triangles:
            edges.add(tuple(sorted((tri[0], tri[1]))))
            edges.add(tuple(sorted((tri[1], tri[2]))))
            edges.add(tuple(sorted((tri[2], tri[0]))))
        
        # 提取边的顶点特征
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        normals = np.asarray(mesh.vertex_normals)
        
        edge_features = []
        edge_list = []
        
        for v1, v2 in edges:
            # 提取两个顶点的特征（坐标 + 法线）
            v1_feat = np.concatenate([vertices[v1], normals[v1]])
            v2_feat = np.concatenate([vertices[v2], normals[v2]])
            
            edge_features.append({
                'v1_feat': v1_feat,
                'v2_feat': v2_feat,
                'edge': (v1, v2)
            })
            edge_list.append((v1, v2))
        
        return edge_features, edge_list
    
    def predict_vertex_importance(self, mesh):
        """预测顶点的特征重要性"""
        if not self.feature_net:
            raise ValueError("Feature network not set")
        
        # 提取顶点特征
        features = self.extract_vertex_features(mesh)
        
        # 转换为PyTorch张量
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # 预测重要性
        with torch.no_grad():
            importance_scores = self.feature_net(features_tensor)
        
        # 转换为numpy数组
        importance_scores = importance_scores.cpu().numpy().flatten()
        
        # 归一化到[0, 1]范围
        if importance_scores.max() > importance_scores.min():
            importance_scores = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min())
        
        return importance_scores
    
    def predict_edge_importance(self, mesh):
        """预测边的特征重要性"""
        if not self.feature_net:
            raise ValueError("Feature network not set")
        
        # 提取边特征
        edge_features, edge_list = self.extract_edge_features(mesh)
        
        if not edge_features:
            return {}, []
        
        # 准备批量数据
        v1_feats = torch.tensor([ef['v1_feat'] for ef in edge_features], dtype=torch.float32).to(self.device)
        v2_feats = torch.tensor([ef['v2_feat'] for ef in edge_features], dtype=torch.float32).to(self.device)
        
        # 预测重要性
        with torch.no_grad():
            importance_scores = self.feature_net(v1_feats, v2_feats)
        
        # 转换为numpy数组
        importance_scores = importance_scores.cpu().numpy().flatten()
        
        # 构建边到重要性的映射
        edge_importance_map = {}
        for i, edge in enumerate(edge_list):
            edge_importance_map[edge] = importance_scores[i]
            # 同时保存反向边
            edge_importance_map[(edge[1], edge[0])] = importance_scores[i]
        
        return edge_importance_map, edge_list
    
    def integrate_features_with_qem(self, mesh, qem_calculator, alpha=0.5):
        """将深度学习特征与QEM算法集成"""
        # 预测顶点重要性
        vertex_importance = self.predict_vertex_importance(mesh)
        
        # 计算原始QEM边成本
        quadrics = qem_calculator.compute_all_quadrics(mesh)
        
        # 构建边集合
        triangles = np.asarray(mesh.triangles)
        edges = set()
        for tri in triangles:
            edges.add(tuple(sorted((tri[0], tri[1]))))
            edges.add(tuple(sorted((tri[1], tri[2]))))
            edges.add(tuple(sorted((tri[2], tri[0]))))
        
        # 计算集成成本
        integrated_edge_costs = []
        vertices = np.asarray(mesh.vertices)
        
        for v1, v2 in edges:
            # 计算QEM成本
            Q = quadrics[v1] + quadrics[v2]
            
            # 尝试三个可能的目标点
            v1_pos = vertices[v1]
            v2_pos = vertices[v2]
            mid_pos = (v1_pos + v2_pos) / 2
            
            # 计算每个目标点的误差
            def compute_error(pos):
                v = np.append(pos, 1.0)
                return np.dot(v, np.dot(Q, v))
            
            e1 = compute_error(v1_pos)
            e2 = compute_error(v2_pos)
            e3 = compute_error(mid_pos)
            
            # 选择最小误差
            min_qem_error = min(e1, e2, e3)
            
            # 确定最佳目标点
            if min_qem_error == e1:
                target_pos = v1_pos
            elif min_qem_error == e2:
                target_pos = v2_pos
            else:
                target_pos = mid_pos
            
            # 获取两个顶点的重要性
            importance1 = vertex_importance[v1]
            importance2 = vertex_importance[v2]
            avg_importance = (importance1 + importance2) / 2
            
            # 集成成本：Cost = QEM_cost * (1 - alpha * importance)
            # 重要性越高，成本越低，越不容易被折叠
            integrated_cost = min_qem_error * (1 - alpha * avg_importance)
            
            integrated_edge_costs.append({
                'edge': (v1, v2),
                'cost': integrated_cost,
                'qem_cost': min_qem_error,
                'importance': avg_importance,
                'target': target_pos
            })
        
        return integrated_edge_costs
    
    def visualize_importance(self, mesh, importance_scores, color_map='viridis'):
        """可视化顶点重要性"""
        import matplotlib.cm as cm
        
        # 确保重要性分数在[0, 1]范围内
        normalized_scores = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min() + 1e-8)
        
        # 使用颜色映射生成顶点颜色
        cmap = cm.get_cmap(color_map)
        vertex_colors = cmap(normalized_scores)[:, :3]  # 去除alpha通道
        
        # 创建可视化网格
        vis_mesh = mesh.deep_copy()
        vis_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        
        return vis_mesh