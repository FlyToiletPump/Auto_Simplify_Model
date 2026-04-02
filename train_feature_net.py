#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练特征提取网络的脚本
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from Data_Prep.data_processor import DataProcessor
from Neural_Modules.train import main as train_main

def generate_data():
    """生成训练数据"""
    print("=== 生成训练数据 ===")
    processor = DataProcessor()
    
    # 生成合成数据
    data_path = processor.generate_synthetic_data(num_samples=10000)
    print(f"合成数据生成完成: {data_path}")
    return data_path

def train_model(data_path):
    """训练模型"""
    print("\n=== 训练特征提取网络 ===")
    
    # 设置环境变量，传递数据路径
    os.environ['TRAIN_DATA_PATH'] = data_path
    
    # 确保Models目录存在
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "Models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models目录创建成功: {models_dir}")
    
    # 调用训练函数
    train_main()

def main():
    """主函数"""
    print("=== 训练特征提取网络 ===")
    
    # 生成数据
    data_path = generate_data()
    
    # 训练模型
    train_model(data_path)
    
    print("\n=== 训练完成 ===")

if __name__ == "__main__":
    main()
