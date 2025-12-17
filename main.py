#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Auto_Simplify_Model - 自动3D网格简化系统主入口

这个系统实现了基于特征感知的3D网格简化算法，结合传统QEM和深度学习技术，
能够在保持重要特征的同时有效减少网格复杂度。
"""

import argparse
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入主要模块
from Data_Prep.data_loader import load_mesh, save_mesh
from Data_Prep.mesh_cleaner import preprocess_mesh
from Core_Algorithms.progressive_lod import ProgressiveLOD
from Evaluation.metrics import MeshQualityEvaluator

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Auto_Simplify_Model - 自动3D网格简化系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    python main.py --input bunny.obj --output simplified_bunny.ply --target 1000
    python main.py --input model.obj --output lods/ --levels 5000 2000 1000 500
    python main.py --input model.obj --output simplified.ply --feature-aware --target 2000
        """
    )
    
    parser.add_argument("--input", "-i", type=str, required=True, 
                      help="输入网格文件路径")
    parser.add_argument("--output", "-o", type=str, required=True, 
                      help="输出文件或目录路径")
    parser.add_argument("--target", "-t", type=int, default=1000, 
                      help="目标面数（单个简化级别）")
    parser.add_argument("--levels", "-l", type=int, nargs="+", 
                      help="多个目标面数（生成渐进式LOD）")
    parser.add_argument("--feature-aware", "-f", action="store_true", 
                      help="使用特征感知简化算法")
    parser.add_argument("--angle-threshold", "-a", type=float, default=45.0, 
                      help="折痕检测的二面角阈值（度数）")
    parser.add_argument("--sample-points", "-s", type=int, default=10000, 
                      help="评估指标计算的采样点数量")
    parser.add_argument("--evaluate", "-e", action="store_true", 
                      help="评估简化质量")
    
    args = parser.parse_args()
    
    print("=== Auto_Simplify_Model - 自动3D网格简化系统 ===")
    
    try:
        # 1. 加载和预处理输入网格
        print(f"\n1. 加载输入网格: {args.input}")
        input_mesh = load_mesh(args.input)
        print(f"原始网格: {len(input_mesh.vertices)} 顶点, {len(input_mesh.triangles)} 三角形")
        
        print("\n2. 预处理网格...")
        preprocessed_mesh = preprocess_mesh(input_mesh)
        print(f"预处理完成: {len(preprocessed_mesh.vertices)} 顶点, {len(preprocessed_mesh.triangles)} 三角形")
        
        # 2. 执行网格简化
        print("\n3. 执行网格简化...")
        
        lod_generator = ProgressiveLOD()
        
        if args.levels:
            # 生成渐进式LOD
            print(f"生成渐进式LOD，目标面数: {args.levels}")
            
            lod_levels = lod_generator.generate_lods(
                mesh=preprocessed_mesh,
                target_faces_list=args.levels,
                feature_aware=args.feature_aware,
                dihedral_angle_threshold=args.angle_threshold
            )
            
            # 创建输出目录（如果需要）
            if not os.path.exists(args.output):
                os.makedirs(args.output)
            
            # 保存所有LOD级别
            for i, (target_faces, lod_mesh) in enumerate(zip(args.levels, lod_levels)):
                output_file = os.path.join(args.output, f"lod_{target_faces}.ply")
                save_mesh(lod_mesh, output_file)
                print(f"LOD {i+1} 保存到: {output_file} ({len(lod_mesh.triangles)} 三角形)")
                
                # 如果需要评估
                if args.evaluate:
                    evaluator = MeshQualityEvaluator(sample_points=args.sample_points)
                    metrics = evaluator.evaluate_all(preprocessed_mesh, lod_mesh)
                    evaluator.print_metrics(metrics)
                    
                    # 保存评估结果
                    metrics_file = os.path.join(args.output, f"lod_{target_faces}_metrics.json")
                    evaluator.save_metrics(metrics, metrics_file)
                    print(f"评估指标保存到: {metrics_file}")
        else:
            # 单个目标面数简化
            print(f"简化到目标面数: {args.target}")
            
            lod_levels = lod_generator.generate_lods(
                mesh=preprocessed_mesh,
                target_faces_list=[args.target],
                feature_aware=args.feature_aware,
                dihedral_angle_threshold=args.angle_threshold
            )
            
            simplified_mesh = lod_levels[0]
            
            # 保存简化结果
            save_mesh(simplified_mesh, args.output)
            print(f"简化结果保存到: {args.output}")
            print(f"简化结果: {len(simplified_mesh.vertices)} 顶点, {len(simplified_mesh.triangles)} 三角形")
            
            # 如果需要评估
            if args.evaluate:
                evaluator = MeshQualityEvaluator(sample_points=args.sample_points)
                metrics = evaluator.evaluate_all(preprocessed_mesh, simplified_mesh)
                evaluator.print_metrics(metrics)
                
                # 保存评估结果
                metrics_file = os.path.splitext(args.output)[0] + "_metrics.json"
                evaluator.save_metrics(metrics, metrics_file)
                print(f"评估指标保存到: {metrics_file}")
        
        print("\n=== 网格简化完成 ===")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请检查输入参数和文件路径是否正确。")
        sys.exit(1)

if __name__ == "__main__":
    main()