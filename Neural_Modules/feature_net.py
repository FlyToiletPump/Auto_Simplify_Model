import torch
import torch.nn as nn


class MeshFeatureNet(nn.Module):
    """轻量级网格特征提取网络"""
    
    def __init__(self, input_dim=6, hidden_dim=32, output_dim=1):
        """
        参数:
            input_dim: 输入特征维度（默认6：3D坐标 + 3D法线）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（默认1：特征重要性分数）
        """
        super(MeshFeatureNet, self).__init__()
        
        # 多层感知器网络
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()  # 输出特征重要性分数 [0, 1]
        )
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)

class LocalFeatureNet(nn.Module):
    """考虑局部邻域的特征提取网络"""
    
    def __init__(self, input_dim=6, hidden_dim=32, output_dim=1, neighborhood_size=5):
        """
        参数:
            input_dim: 单个顶点的输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            neighborhood_size: 局部邻域的顶点数量
        """
        super(LocalFeatureNet, self).__init__()
        
        self.neighborhood_size = neighborhood_size
        total_input_dim = input_dim * neighborhood_size
        
        # 共享的邻域特征提取器
        self.neighborhood_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 全局特征融合器
        self.global_fusion = nn.Sequential(
            nn.Linear(hidden_dim * neighborhood_size, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """前向传播
        x: [batch_size, neighborhood_size, input_dim]
        """
        batch_size, _, _ = x.shape
        
        # 对每个邻域顶点进行编码
        encoded_neighbors = []
        for i in range(self.neighborhood_size):
            neighbor_features = x[:, i, :]
            encoded = self.neighborhood_encoder(neighbor_features)
            encoded_neighbors.append(encoded)
        
        # 连接所有邻域的编码特征
        combined_features = torch.cat(encoded_neighbors, dim=1)
        
        # 融合邻域特征并输出
        output = self.global_fusion(combined_features)
        
        return output

class EdgeFeatureNet(nn.Module):
    """边特征提取网络"""
    
    def __init__(self, vertex_feature_dim=6, hidden_dim=32, output_dim=1):
        """
        参数:
            vertex_feature_dim: 单个顶点的特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        super(EdgeFeatureNet, self).__init__()
        
        # 边特征是两个顶点特征的连接
        edge_input_dim = vertex_feature_dim * 2
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, v1_features, v2_features):
        """前向传播
        v1_features: [batch_size, vertex_feature_dim]
        v2_features: [batch_size, vertex_feature_dim]
        """
        # 连接两个顶点的特征
        edge_features = torch.cat([v1_features, v2_features], dim=1)
        
        # 计算边的重要性分数
        edge_importance = self.edge_encoder(edge_features)
        
        return edge_importance

def create_feature_net(net_type="vertex", **kwargs):
    """创建特征提取网络实例"""
    if net_type == "vertex":
        return MeshFeatureNet(**kwargs)
    elif net_type == "local":
        return LocalFeatureNet(**kwargs)
    elif net_type == "edge":
        return EdgeFeatureNet(**kwargs)
    else:
        raise ValueError(f"Unknown network type: {net_type}")