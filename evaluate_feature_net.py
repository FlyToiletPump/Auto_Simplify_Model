#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估特征提取网络的脚本
"""

import os
import sys
import numpy as np
import h5py
import torch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from Neural_Modules.feature_net import create_feature_net
from Neural_Modules.trainer import FeatureNetTrainer
from Data_Prep.data_processor import DataProcessor

def load_data(file_path):
    """加载评估数据"""
    with h5py.File(file_path, 'r') as f:
        features = f['features'][:]
        importance = f['importance'][:]
    return features, importance

def evaluate_model(model_path, test_data_path):
    """评估模型性能"""
    print(f"=== 评估模型: {model_path} ===")
    
    # 加载测试数据
    print(f"加载测试数据: {test_data_path}")
    features, importance = load_data(test_data_path)
    
    # 检测模型类型
    model_type = "vertex"
    if "edge" in model_path:
        model_type = "edge"
    elif "local" in model_path:
        model_type = "local"
    
    # 创建模型
    model = create_feature_net(model_type)
    
    # 创建训练器并加载模型
    trainer = FeatureNetTrainer(model)
    trainer.load_model(model_path)
    
    # 预测
    print("进行预测...")
    predictions = trainer.predict(features)
    
    # 计算评估指标
    print("计算评估指标...")
    
    # 均方误差
    mse = np.mean((predictions - importance) ** 2)
    print(f"均方误差 (MSE): {mse:.6f}")
    
    # 平均绝对误差
    mae = np.mean(np.abs(predictions - importance))
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    
    # 相关系数
    correlation = np.corrcoef(predictions.flatten(), importance.flatten())[0, 1]
    print(f"相关系数: {correlation:.6f}")
    
    # 计算准确率（预测值与真实值的差异在0.1以内）
    accuracy = np.mean(np.abs(predictions - importance) < 0.1)
    print(f"准确率 (误差<0.1): {accuracy:.6f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'accuracy': accuracy
    }

def main():
    """主函数"""
    print("=== 评估特征提取网络 ===")
    
    # 生成测试数据
    print("生成测试数据...")
    processor = DataProcessor()
    test_data_path = processor.generate_synthetic_data(num_samples=2000, output_file="test_data.h5")
    
    # 评估不同模型
    models_dir = "./Models"
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        print("没有找到模型文件，请先训练模型")
        return
    
    results = {}
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        print(f"\n评估模型: {model_file}")
        result = evaluate_model(model_path, test_data_path)
        results[model_file] = result
    
    # 打印汇总结果
    print("\n=== 评估结果汇总 ===")
    for model_file, result in results.items():
        print(f"\n{model_file}:")
        print(f"  MSE: {result['mse']:.6f}")
        print(f"  MAE: {result['mae']:.6f}")
        print(f"  相关系数: {result['correlation']:.6f}")
        print(f"  准确率: {result['accuracy']:.6f}")

if __name__ == "__main__":
    main()
