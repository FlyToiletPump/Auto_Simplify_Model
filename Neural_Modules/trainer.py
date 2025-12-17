import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


class FeatureNetTrainer:
    """训练特征提取网络"""
    
    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-5):
        """
        参数:
            model: 要训练的特征提取网络
            learning_rate: 学习率
            weight_decay: L2正则化权重
        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 定义优化器和损失函数
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.loss_fn = F.mse_loss  # 均方误差损失
    
    def prepare_data(self, features, labels):
        """准备训练数据"""
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        labels = torch.tensor(labels, dtype=torch.float32).to(self.device)
        return features, labels
    
    def train_step(self, features, labels):
        """单步训练"""
        self.model.train()
        
        # 前向传播
        outputs = self.model(features)
        loss = self.loss_fn(outputs, labels)
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate_step(self, features, labels):
        """单步验证"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(features)
            loss = self.loss_fn(outputs, labels)
        
        return loss.item(), outputs.cpu().numpy()
    
    def train(self, train_features, train_labels, val_features=None, val_labels=None, epochs=50, batch_size=32):
        """完整训练过程"""
        # 准备训练数据
        train_features, train_labels = self.prepare_data(train_features, train_labels)
        
        # 准备验证数据（如果提供）
        has_validation = val_features is not None and val_labels is not None
        if has_validation:
            val_features, val_labels = self.prepare_data(val_features, val_labels)
        
        # 计算训练批次数量
        num_train_batches = (len(train_features) + batch_size - 1) // batch_size
        
        # 训练历史
        train_history = []
        val_history = []
        
        for epoch in range(epochs):
            # 训练阶段
            train_loss = 0.0
            
            # 打乱训练数据
            indices = np.random.permutation(len(train_features))
            train_features = train_features[indices]
            train_labels = train_labels[indices]
            
            # 批量训练
            for batch_idx in range(num_train_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(train_features))
                
                batch_features = train_features[start_idx:end_idx]
                batch_labels = train_labels[start_idx:end_idx]
                
                loss = self.train_step(batch_features, batch_labels)
                train_loss += loss * len(batch_features)
            
            # 计算平均训练损失
            avg_train_loss = train_loss / len(train_features)
            train_history.append(avg_train_loss)
            
            # 验证阶段
            if has_validation:
                val_loss, val_predictions = self.validate_step(val_features, val_labels)
                val_history.append(val_loss)
                
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}")
        
        return train_history, val_history
    
    def predict(self, features):
        """使用训练好的模型进行预测"""
        self.model.eval()
        
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(features)
        
        return predictions.cpu().numpy()
    
    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")

def generate_synthetic_data(num_samples=1000, input_dim=6):
    """生成合成训练数据（用于演示）"""
    # 生成随机顶点特征（坐标 + 法线）
    features = np.random.rand(num_samples, input_dim)
    
    # 生成合成标签（基于某些特征模式）
    # 示例：如果法线的z分量较大，则认为特征更重要
    labels = features[:, 5:6]  # 法线的z分量
    labels = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)
    
    return features, labels

if __name__ == "__main__":
    # 演示训练过程
    from feature_net import MeshFeatureNet
    
    # 创建模型
    model = MeshFeatureNet()
    
    # 创建训练器
    trainer = FeatureNetTrainer(model)
    
    # 生成合成数据
    train_features, train_labels = generate_synthetic_data(1000)
    val_features, val_labels = generate_synthetic_data(200)
    
    # 训练模型
    trainer.train(
        train_features, train_labels,
        val_features, val_labels,
        epochs=20,
        batch_size=32
    )
    
    # 保存模型
    trainer.save_model("feature_net.pth")