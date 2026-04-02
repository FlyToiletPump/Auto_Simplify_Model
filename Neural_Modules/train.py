import os
import argparse
import numpy as np
import h5py
import torch
from .feature_net import create_feature_net
from .trainer import FeatureNetTrainer

def load_data(file_path):
    """加载训练数据"""
    with h5py.File(file_path, 'r') as f:
        features = f['features'][:]
        importance = f['importance'][:]
    return features, importance

def split_data(features, importance, val_split=0.2):
    """分割训练和验证数据"""
    indices = np.random.permutation(len(features))
    split_idx = int(len(features) * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_features = features[train_indices]
    train_importance = importance[train_indices]
    val_features = features[val_indices]
    val_importance = importance[val_indices]
    
    return train_features, train_importance, val_features, val_importance

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练特征提取网络")
    parser.add_argument('--data', type=str, default=os.environ.get('TRAIN_DATA_PATH', '../Data/processed/synthetic_data.h5'), help='训练数据路径')
    parser.add_argument('--model-type', type=str, default='vertex', choices=['vertex', 'local', 'edge'], help='模型类型')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--hidden-dim', type=int, default=32, help='隐藏层维度')
    parser.add_argument('--output-dir', type=str, default='Models', help='模型保存目录')
    
    args = parser.parse_args()
    
    # 创建模型保存目录
    # 如果output_dir是相对路径，转换为项目根目录下的绝对路径
    print(f"原始output_dir: {args.output_dir}")
    print(f"是否是绝对路径: {os.path.isabs(args.output_dir)}")
    
    if not os.path.isabs(args.output_dir):
        # 获取当前文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"当前文件目录: {current_dir}")
        # 项目根目录是当前目录的上一级
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        print(f"项目根目录: {project_root}")
        args.output_dir = os.path.join(project_root, args.output_dir)
        print(f"转换后的output_dir: {args.output_dir}")
    
    print(f"最终output_dir: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"目录创建成功: {os.path.exists(args.output_dir)}")
    
    # 加载数据
    print(f"加载数据: {args.data}")
    features, importance = load_data(args.data)
    
    # 分割数据
    train_features, train_importance, val_features, val_importance = split_data(features, importance)
    print(f"训练数据: {len(train_features)} 样本")
    print(f"验证数据: {len(val_features)} 样本")
    
    # 创建模型
    print(f"创建模型: {args.model_type}")
    model = create_feature_net(args.model_type, hidden_dim=args.hidden_dim)
    
    # 创建训练器
    trainer = FeatureNetTrainer(model, learning_rate=args.lr)
    
    # 训练模型
    print("开始训练...")
    train_history, val_history = trainer.train(
        train_features, train_importance,
        val_features, val_importance,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # 保存模型
    model_name = f"{args.model_type}_feature_net.pth"
    model_path = os.path.join(args.output_dir, model_name)
    trainer.save_model(model_path)
    
    # 保存训练历史
    history_path = os.path.join(args.output_dir, f"{args.model_type}_train_history.npy")
    np.save(history_path, {'train': train_history, 'val': val_history})
    
    print("训练完成！")
    print(f"模型保存到: {model_path}")

if __name__ == "__main__":
    main()
