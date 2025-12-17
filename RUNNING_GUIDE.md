# Auto_Simplify_Model 运行指南

本文档详细介绍了如何运行和使用 Auto_Simplify_Model 自动3D网格简化系统。

## 目录

- [Auto_Simplify_Model 运行指南](#auto_simplify_model-运行指南)
  - [目录](#目录)
  - [1. 项目简介](#1-项目简介)
  - [2. 环境配置](#2-环境配置)
    - [2.1 Python版本要求](#21-python版本要求)
    - [2.2 安装依赖](#22-安装依赖)
  - [3. 快速开始](#3-快速开始)
    - [3.1 单个网格简化](#31-单个网格简化)
    - [3.2 生成渐进式LOD](#32-生成渐进式lod)
    - [3.3 特征感知简化](#33-特征感知简化)
  - [4. 运行示例代码](#4-运行示例代码)
  - [5. 命令行参数详解](#5-命令行参数详解)
  - [6. 功能说明](#6-功能说明)
    - [6.1 数据预处理](#61-数据预处理)
    - [6.2 网格简化算法](#62-网格简化算法)
    - [6.3 质量评估](#63-质量评估)
  - [7. 高级用法](#7-高级用法)
  - [8. 常见问题](#8-常见问题)

## 1. 项目简介

Auto_Simplify_Model 是一个基于特征感知的3D网格简化系统，结合传统QEM（二次误差度量）和深度学习技术，能够在保持重要特征的同时有效减少网格复杂度。

主要功能：
- 单级网格简化
- 渐进式LOD（细节层次）生成
- 特征感知简化（保护折痕边和角点）
- 简化质量评估
- 支持多种3D模型格式

## 2. 环境配置

### 2.1 Python版本要求

推荐使用 **Python 3.9**（完全兼容所有依赖）

### 2.2 安装依赖

1. 创建虚拟环境（推荐）

   ```bash
   # 使用conda创建虚拟环境
   conda create -n auto_simplify python=3.9
   conda activate auto_simplify
   
   # 或使用venv创建虚拟环境
   python -m venv auto_simplify_env
   # Windows: .\auto_simplify_env\Scripts\activate
   # Linux/Mac: source auto_simplify_env/bin/activate
   ```

2. 安装依赖包

   ```bash
   cd Auto_Simplify_Model
   python -m pip install -r requirements.txt
   ```

## 3. 快速开始

### 3.1 单个网格简化

将输入网格简化到指定目标面数：

```bash
python main.py --input bunny.obj --output simplified_bunny.ply --target 1000
```

参数说明：
- `--input`/`-i`: 输入网格文件路径
- `--output`/`-o`: 输出简化后的网格文件路径
- `--target`/`-t`: 目标面数（默认1000）

### 3.2 生成渐进式LOD

生成多个不同细节层次的简化模型：

```bash
python main.py --input model.obj --output lods/ --levels 5000 2000 1000 500
```

参数说明：
- `--levels`/`-l`: 多个目标面数列表（按降序排列）
- 输出目录会自动创建，保存所有LOD级别

### 3.3 特征感知简化

使用特征感知算法保护关键特征（折痕边和角点）：

```bash
python main.py --input model.obj --output simplified.ply --feature-aware --target 2000
```

参数说明：
- `--feature-aware`/`-f`: 启用特征感知简化
- `--angle-threshold`/`-a`: 折痕检测的二面角阈值（默认45度）

## 4. 运行示例代码

项目提供了一个完整的示例脚本，演示所有主要功能：

```bash
cd Examples
python basic_simplification.py
```

示例功能包括：
1. 加载和预处理网格
2. 检测折痕边
3. 基本QEM简化
4. 特征感知QEM简化
5. 渐进式LOD生成
6. 质量评估
7. 结果可视化

如果存在示例模型 `../QEM_MVP/bunny_10k.obj`，脚本会使用该模型；否则会创建一个立方体用于演示。

## 5. 命令行参数详解

```bash
python main.py --help
```

完整参数列表：

| 参数 | 缩写 | 类型 | 必填 | 描述 |
|------|------|------|------|------|
| `--input` | `-i` | 字符串 | 是 | 输入网格文件路径 |
| `--output` | `-o` | 字符串 | 是 | 输出文件或目录路径 |
| `--target` | `-t` | 整数 | 否 | 目标面数（单个简化级别，默认1000） |
| `--levels` | `-l` | 整数列表 | 否 | 多个目标面数（生成渐进式LOD） |
| `--feature-aware` | `-f` | 标志 | 否 | 使用特征感知简化算法 |
| `--angle-threshold` | `-a` | 浮点数 | 否 | 折痕检测的二面角阈值（度数，默认45.0） |
| `--sample-points` | `-s` | 整数 | 否 | 评估指标计算的采样点数量（默认10000） |
| `--evaluate` | `-e` | 标志 | 否 | 评估简化质量 |

## 6. 功能说明

### 6.1 数据预处理

系统会自动对输入网格进行以下预处理：
- 去重顶点
- 修复非流形边
- 移除退化三角形
- 归一化到单位球
- 去除噪声（可选）

### 6.2 网格简化算法

系统支持两种简化算法：

1. **基本QEM算法**：
   - 基于二次误差度量
   - 快速高效
   - 适合一般简化需求

2. **特征感知QEM算法**：
   - 扩展基本QEM
   - 通过二面角检测折痕边
   - 为关键特征分配更高权重
   - 保持模型细节和视觉质量

### 6.3 质量评估

使用 `--evaluate` 参数可以评估简化质量，生成以下指标：
- 几何精度：Hausdorff距离、RMS误差
- 特征保留：折痕保留率
- 视觉一致性：法线一致性
- 简化率：顶点/面数减少比例

## 7. 高级用法

### 评估简化质量

```bash
python main.py --input model.obj --output simplified.ply --target 1000 --evaluate
```

会生成一个 `simplified_metrics.json` 文件，包含详细的评估指标。

### 自定义折痕检测阈值

```bash
python main.py --input model.obj --output simplified.ply --feature-aware --angle-threshold 30 --target 2000
```

较低的阈值会检测出更多的折痕边，提供更好的特征保护。

## 8. 常见问题

1. **问题**：python解释器和依赖项目的版本冲突
   **解决方案**：保持python解释器的版本为3.8-3.10。

2. **问题**：无法加载某些3D模型格式
   **解决方案**：系统主要支持.obj、.ply、.stl等常见格式，确保您的模型格式正确。

3. **问题**：简化过程非常慢
   **解决方案**：
   - 对于大型模型，建议增加系统内存
   - 减少目标面数可以加快处理速度
   - 关闭特征感知功能（--feature-aware）也可以提高速度

4. **问题**：可视化窗口无响应
   **解决方案**：
   - 按ESC键退出可视化
   - 如果仍然无响应，可以终止程序并重新运行
   - 降低模型复杂度或采样点数量

如果您遇到其他问题，请检查输入参数和文件路径是否正确，或者查看系统输出的错误信息。