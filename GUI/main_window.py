#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主窗口实现
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .parameter_panel import ParameterPanel
from .model_preview import ModelPreview
from Core_Algorithms.progressive_lod import ProgressiveLOD
from Data_Prep.mesh_cleaner import clean_mesh

class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("网格简化工具")
        self.root.geometry("1000x700")
        
        # 变量
        self.input_model_path = tk.StringVar()
        self.output_model_path = tk.StringVar()
        self.target_faces = tk.IntVar(value=1000)
        self.use_open3d = tk.BooleanVar(value=False)
        self.use_deep_learning = tk.BooleanVar(value=False)
        self.model_path = tk.StringVar(value="Models/vertex_feature_net.pth")
        
        # 布局
        self.create_widgets()
    
    def create_widgets(self):
        # 顶部文件选择区域
        file_frame = tk.Frame(self.root, padx=10, pady=10)
        file_frame.pack(fill=tk.X)
        
        tk.Label(file_frame, text="输入模型:").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(file_frame, textvariable=self.input_model_path, width=50).grid(row=0, column=1, padx=5)
        tk.Button(file_frame, text="浏览", command=self.browse_input).grid(row=0, column=2, padx=5)
        
        tk.Label(file_frame, text="输出目录:").grid(row=1, column=0, sticky=tk.W)
        tk.Entry(file_frame, textvariable=self.output_model_path, width=50).grid(row=1, column=1, padx=5)
        tk.Button(file_frame, text="浏览", command=self.browse_output).grid(row=1, column=2, padx=5)
        
        # 参数设置面板
        param_panel = ParameterPanel(self.root, self.target_faces, self.use_open3d, 
                                     self.use_deep_learning, self.model_path)
        param_panel.pack(fill=tk.X, padx=10, pady=5)
        
        # 模型预览区域
        preview_frame = tk.Frame(self.root, padx=10, pady=10)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        self.preview = ModelPreview(preview_frame)
        
        # 底部按钮区域
        button_frame = tk.Frame(self.root, padx=10, pady=10)
        button_frame.pack(fill=tk.X)
        
        tk.Button(button_frame, text="加载模型", command=self.load_model).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="执行减面", command=self.simplify_model).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="保存结果", command=self.save_model).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="退出", command=self.root.quit).pack(side=tk.RIGHT, padx=5)
    
    def browse_input(self):
        filename = filedialog.askopenfilename(
            filetypes=[("3D模型文件", "*.obj *.ply *.stl *.off"), ("所有文件", "*.*")]
        )
        if filename:
            self.input_model_path.set(filename)
    
    def browse_output(self):
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_model_path.set(directory)
    
    def load_model(self):
        model_path = self.input_model_path.get()
        if not model_path:
            messagebox.showerror("错误", "请选择输入模型")
            return
        
        try:
            self.preview.load_model(model_path)
            messagebox.showinfo("成功", "模型加载成功")
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {str(e)}")
    
    def simplify_model(self):
        model_path = self.input_model_path.get()
        if not model_path:
            messagebox.showerror("错误", "请选择输入模型")
            return
        
        target_faces = self.target_faces.get()
        if target_faces <= 0:
            messagebox.showerror("错误", "目标面数必须大于0")
            return
        
        try:
            # 加载模型
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(model_path)
            
            # 预处理
            cleaned_mesh = clean_mesh(mesh)
            
            # 执行减面
            lod_generator = ProgressiveLOD()
            
            # 执行简化
            simplified_mesh = lod_generator.generate_lods(
                cleaned_mesh,
                target_faces_list=[target_faces],
                feature_aware=self.use_deep_learning.get(),
                use_open3d=self.use_open3d.get()
            )[0]
            
            # 显示结果
            self.preview.show_simplified(simplified_mesh)
            
            # 构建详细的成功消息
            success_message = f"减面完成，目标面数: {target_faces}"
            if self.use_open3d.get():
                success_message += "\n使用了Open3D内置简化方法"
            if self.use_deep_learning.get():
                success_message += "\n使用了深度学习特征"
            
            messagebox.showinfo("成功", success_message)
            
            # 保存简化后的模型到临时变量
            self.simplified_mesh = simplified_mesh
            
        except Exception as e:
            messagebox.showerror("错误", f"减面失败: {str(e)}")
    
    def save_model(self):
        if not hasattr(self, 'simplified_mesh'):
            messagebox.showerror("错误", "请先执行减面")
            return
        
        output_dir = self.output_model_path.get()
        if not output_dir:
            messagebox.showerror("错误", "请选择输出路径")
            return
        
        try:
            import open3d as o3d
            import os
            
            # 从输入文件名中提取基本名称
            input_path = self.input_model_path.get()
            if input_path:
                base_name = os.path.splitext(os.path.basename(input_path))[0]
            else:
                base_name = "simplified_model"
            
            # 生成输出文件名
            output_filename = f"{base_name}_simplified.obj"
            output_path = os.path.join(output_dir, output_filename)
            
            o3d.io.write_triangle_mesh(output_path, self.simplified_mesh)
            messagebox.showinfo("成功", f"模型保存成功: {output_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存模型失败: {str(e)}")
    
    def run(self):
        self.root.mainloop()
