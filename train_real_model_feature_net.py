#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用真实3D模型数据训练特征提取网络
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from Data_Prep.real_model_loader import get_dataloader
from Neural_Modules.feature_net import create_feature_net


def train_real_model_feature_net():
    """使用真实3D模型数据训练特征提取网络"""
    # 配置参数
    config = {
        "data_dir": "e:\\GP\\Auto_Simplify_Model-main\\Auto_Simplify_Model\\Thingi10K",
        "batch_size": 8,  # 减少批次大小，避免内存问题
        "num_workers": 4,  # 设置为0避免Windows多进程崩溃问题
        "learning_rate": 1e-2,  # 增加初始学习率，加快收敛
        "epochs": 10,  # 减少训练轮次
        "model_save_path": "Models\\real_model_feature_net2.pth",
        "history_save_path": "Data\\real_model_train_history.npy",
        "network_type": "vertex",  # vertex, local, edge
        "hidden_dim": 16,  # 简化模型结构
        "use_amp": True,  # 使用混合精度训练
        "sample_rate": 0.2,  # 只使用10%的数据
        "max_vertices": 5000  # 限制每个模型的顶点数，减少内存使用
    }

    # 创建保存目录
    os.makedirs(os.path.dirname(config["model_save_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["history_save_path"]), exist_ok=True)

    # 加载数据
    print("Loading real 3D model data...")
    dataloader = get_dataloader(
        config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        max_vertices=config["max_vertices"]
    )

    # 创建模型
    print("Creating feature extraction network...")
    model = create_feature_net(
        net_type=config["network_type"],
        input_dim=6,  # 3D坐标 + 3D法线
        hidden_dim=config["hidden_dim"],
        output_dim=1
    )

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if config["use_amp"] and device.type == "cuda" else None

    # 训练历史
    history = {
        "loss": [],
        "val_loss": []
    }

    # 训练循环
    print("Starting training...")
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        batch_count = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for features_list, labels_list in progress_bar:
            # 跳过空批次
            if not features_list or not labels_list:
                continue

            # 数据采样
            if np.random.random() > config["sample_rate"]:
                continue

            # 处理每个模型
            for features, labels in zip(features_list, labels_list):
                # 跳过空数据
                if features.shape[0] == 0 or labels.shape[0] == 0:
                    continue

                # 移动到设备
                features = features.to(device)
                labels = labels.to(device)

                # 混合精度训练
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(features)
                        loss = criterion(outputs.squeeze(), labels)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(features)
                    loss = criterion(outputs.squeeze(), labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                batch_count += 1

                if batch_count > 0:
                    progress_bar.set_postfix({"loss": f"{running_loss / batch_count:.4f}"})

        # 计算平均损失
        epoch_loss = running_loss / batch_count if batch_count > 0 else 0
        history["loss"].append(epoch_loss)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        if batch_count > 0:
            scheduler.step(epoch_loss)

    # 保存模型和历史
    torch.save(model.state_dict(), config["model_save_path"])
    np.save(config["history_save_path"], history)

    print(f"Training completed!")
    print(f"Model saved to: {config['model_save_path']}")
    print(f"Training history saved to: {config['history_save_path']}")

    return model


def evaluate_real_model_feature_net():
    """评估训练好的特征提取网络"""
    # 配置参数
    config = {
        "data_dir": "e:\\GP\\Auto_Simplify_Model-main\\Auto_Simplify_Model\\Thingi10K",
        "batch_size": 4,
        "num_workers": 0,
        "model_path": "Models\\real_model_feature_net2.pth",
        "network_type": "vertex",
        "hidden_dim": 16
    }

    # 加载数据
    print("Loading evaluation data...")
    dataloader = get_dataloader(
        config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

    # 创建模型并加载权重
    print("Loading trained model...")
    model = create_feature_net(
        net_type=config["network_type"],
        input_dim=6,
        hidden_dim=config["hidden_dim"],
        output_dim=1
    )

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 加载权重
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.eval()

    # 评估
    total_loss = 0.0
    criterion = nn.MSELoss()

    print("Evaluating model...")
    with torch.no_grad():
        for features_list, labels_list in tqdm(dataloader):
            for features, labels in zip(features_list, labels_list):
                if features.shape[0] == 0 or labels.shape[0] == 0:
                    continue
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels)
                total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Loss: {avg_loss:.4f}")

    return avg_loss


if __name__ == "__main__":
    # 训练模型
    train_real_model_feature_net()

    # 评估模型
    evaluate_real_model_feature_net()