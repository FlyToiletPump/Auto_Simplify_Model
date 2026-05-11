#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主窗口实现
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys
import threading
import time

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
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # 变量
        self.input_model_path = tk.StringVar()
        self.output_model_path = tk.StringVar()
        self.target_faces = tk.IntVar(value=1000)
        self.use_open3d = tk.BooleanVar(value=False)
        self.use_deep_learning = tk.BooleanVar(value=False)
        self.model_path = tk.StringVar(value="Models/vertex_feature_net.pth")
        self.lod_levels = tk.StringVar(value="1000,2000,5000")  # 多级LOD设置
        self.current_progress = tk.DoubleVar(value=0)
        self.status_message = tk.StringVar(value="就绪")
        
        # 布局
        self.create_menu()
        self.create_widgets()
    
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开模型", command=self.browse_input)
        file_menu.add_command(label="保存模型", command=self.save_model)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 功能菜单
        feature_menu = tk.Menu(menubar, tearoff=0)
        feature_menu.add_command(label="加载模型", command=self.load_model)
        feature_menu.add_command(label="执行减面", command=self.simplify_model)
        feature_menu.add_command(label="生成多级LOD", command=self.generate_multiple_lods)
        menubar.add_cascade(label="功能", menu=feature_menu)
        
        # 工具菜单
        tool_menu = tk.Menu(menubar, tearoff=0)
        tool_menu.add_command(label="清理模型", command=self.clean_model)
        tool_menu.add_command(label="评估模型", command=self.evaluate_model)
        menubar.add_cascade(label="工具", menu=tool_menu)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self.show_about)
        help_menu.add_command(label="使用说明", command=self.show_help)
        menubar.add_cascade(label="帮助", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_widgets(self):
        # 顶部文件选择区域
        file_frame = tk.Frame(self.root, padx=10, pady=10)
        file_frame.pack(fill=tk.X)
        
        tk.Label(file_frame, text="输入模型:").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(file_frame, textvariable=self.input_model_path, width=60).grid(row=0, column=1, padx=5)
        tk.Button(file_frame, text="浏览", command=self.browse_input).grid(row=0, column=2, padx=5)
        
        tk.Label(file_frame, text="输出目录:").grid(row=1, column=0, sticky=tk.W)
        tk.Entry(file_frame, textvariable=self.output_model_path, width=60).grid(row=1, column=1, padx=5)
        tk.Button(file_frame, text="浏览", command=self.browse_output).grid(row=1, column=2, padx=5)
        
        # 参数设置面板
        param_panel = ParameterPanel(self.root, self.target_faces, self.use_open3d, 
                                     self.use_deep_learning, self.model_path, self.lod_levels)
        param_panel.pack(fill=tk.X, padx=10, pady=5)
        
        # 主内容区域
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左半部分：模型预览
        preview_frame = tk.Frame(main_frame, width=800)
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.preview = ModelPreview(preview_frame)
        
        # 右半部分：信息与评估
        info_frame = tk.Frame(main_frame, width=400, relief=tk.RAISED, borderwidth=1)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        
        # 进度条
        progress_frame = tk.Frame(info_frame, padx=10, pady=5)
        progress_frame.pack(fill=tk.X)
        
        tk.Label(progress_frame, text="进度:").pack(side=tk.LEFT)
        ttk.Progressbar(progress_frame, variable=self.current_progress, length=200, mode='determinate').pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # 状态信息
        status_frame = tk.Frame(info_frame, padx=10, pady=5)
        status_frame.pack(fill=tk.X)
        
        tk.Label(status_frame, text="状态:").pack(side=tk.LEFT)
        tk.Label(status_frame, textvariable=self.status_message, fg="blue").pack(side=tk.LEFT, padx=10)
        
        # 评估报告区
        eval_frame = tk.LabelFrame(info_frame, text="评估报告", padx=10, pady=10)
        eval_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.eval_text = tk.Text(eval_frame, height=10, wrap=tk.WORD)
        self.eval_text.pack(fill=tk.BOTH, expand=True)
        self.eval_text.insert(tk.END, "评估报告将在此显示...")
        
        # 底部按钮区域
        button_frame = tk.Frame(self.root, padx=10, pady=10)
        button_frame.pack(fill=tk.X)
        
        tk.Button(button_frame, text="加载模型", command=self.load_model).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="执行减面", command=self.simplify_model).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="生成多级LOD", command=self.generate_multiple_lods).pack(side=tk.LEFT, padx=5)
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
        
        def load_thread():
            try:
                self.root.after(0, lambda: self.status_message.set("加载模型中..."))
                
                import open3d as o3d
                mesh = o3d.io.read_triangle_mesh(model_path)
                
                vertices_count = len(mesh.vertices)
                triangles_count = len(mesh.triangles)
                
                # 保存mesh对象，避免重复加载
                self.original_mesh = mesh
                
                # 更新预览区域的模型信息
                self.root.after(0, lambda: self.preview.set_original_info(vertices_count, triangles_count))
                self.root.after(0, lambda: self.preview.show_3d_model_async(mesh, "原始模型", "original"))
                self.root.after(0, lambda: self.status_message.set(f"模型加载成功 - {vertices_count} 顶点, {triangles_count} 面"))
                
            except Exception as e:
                error_msg = str(e)
                def show_error():
                    self.status_message.set("加载模型失败")
                    messagebox.showerror("错误", f"加载模型失败: {error_msg}")
                self.root.after(0, show_error)
        
        thread = threading.Thread(target=load_thread)
        thread.daemon = True
        thread.start()
    
    def simplify_model(self):
        input_model_path = self.input_model_path.get()
        if not input_model_path:
            messagebox.showerror("错误", "请选择输入模型")
            return
        
        target_faces = self.target_faces.get()
        if target_faces <= 0:
            messagebox.showerror("错误", "目标面数必须大于0")
            return
        
        # 启动线程执行减面
        def simplify_thread():
            try:
                # 立即更新状态
                self.root.after(0, lambda: self.status_message.set("执行减面中..."))
                self.root.after(0, lambda: self.current_progress.set(0))
                self.root.update_idletasks()
                
                # 加载模型
                import open3d as o3d
                mesh = o3d.io.read_triangle_mesh(input_model_path)
                
                # 更新进度
                self.root.after(0, lambda: self.current_progress.set(20))
                self.root.update_idletasks()
                
                # 预处理
                cleaned_mesh = clean_mesh(mesh)
                
                # 更新进度
                self.root.after(0, lambda: self.current_progress.set(40))
                self.root.update_idletasks()
                
                # 执行减面
                lod_generator = ProgressiveLOD()
                
                # 准备特征提取器（如果使用深度学习）
                feature_extractor = None
                if self.use_deep_learning.get():
                    from Neural_Modules.feature_integrator import FeatureIntegrator
                    feature_model_path = self.model_path.get()
                    try:
                        feature_extractor = FeatureIntegrator(model_path=feature_model_path)
                        print(f"加载深度学习模型成功: {feature_model_path}")
                    except Exception as e:
                        error_msg = str(e)
                        def show_dl_error():
                            messagebox.showerror("错误", f"加载深度学习模型失败: {error_msg}")
                        self.root.after(0, show_dl_error)
                        return
                
                # 更新进度
                self.root.after(0, lambda: self.current_progress.set(60))
                self.root.update_idletasks()
                
                # 执行简化
                simplified_mesh = lod_generator.generate_lods(
                    cleaned_mesh,
                    target_faces_list=[target_faces],
                    feature_aware=self.use_deep_learning.get(),
                    use_open3d=self.use_open3d.get(),
                    feature_extractor=feature_extractor
                )[0]
                
                # 更新进度
                self.root.after(0, lambda: self.current_progress.set(80))
                self.root.update_idletasks()
                
                # 显示结果
                self.preview.show_simplified(simplified_mesh)
                
                # 构建详细的成功消息
                success_message = f"减面完成，目标面数: {target_faces}"
                if self.use_open3d.get():
                    success_message += "\n使用了Open3D内置简化方法"
                if self.use_deep_learning.get():
                    success_message += "\n使用了深度学习特征"
                
                # 保存简化后的模型到临时变量
                self.simplified_mesh = simplified_mesh
                
                # 生成评估报告
                self.generate_evaluation_report(mesh, simplified_mesh)
                
                # 完成后立即更新状态
                def show_success():
                    self.current_progress.set(100)
                    self.status_message.set("减面完成")
                    messagebox.showinfo("成功", success_message)
                self.root.after(0, show_success)
                
            except Exception as e:
                error_msg = str(e)
                def show_error():
                    self.status_message.set("减面失败")
                    messagebox.showerror("错误", f"减面失败: {error_msg}")
                self.root.after(0, show_error)
        
        # 启动线程
        thread = threading.Thread(target=simplify_thread)
        thread.daemon = True
        thread.start()
    
    def generate_multiple_lods(self):
        input_model_path = self.input_model_path.get()
        if not input_model_path:
            messagebox.showerror("错误", "请选择输入模型")
            return
        
        # 解析LOD级别
        lod_levels_str = self.lod_levels.get()
        try:
            lod_levels = [int(level.strip()) for level in lod_levels_str.split(",")]
            if not lod_levels or any(level <= 0 for level in lod_levels):
                raise ValueError("LOD级别必须是正整数")
        except ValueError as e:
            messagebox.showerror("错误", f"LOD级别格式错误: {str(e)}")
            return
        
        # 启动线程执行多级LOD生成
        def lod_thread():
            try:
                # 立即更新状态
                self.root.after(0, lambda: self.status_message.set("生成多级LOD中..."))
                self.root.after(0, lambda: self.current_progress.set(0))
                
                # 加载模型
                import open3d as o3d
                mesh = o3d.io.read_triangle_mesh(input_model_path)
                
                # 更新进度
                self.root.after(0, lambda: self.current_progress.set(20))
                
                # 预处理
                cleaned_mesh = clean_mesh(mesh)
                
                # 更新进度
                self.root.after(0, lambda: self.current_progress.set(30))
                
                # 执行减面
                lod_generator = ProgressiveLOD()
                
                # 准备特征提取器（如果使用深度学习）
                feature_extractor = None
                if self.use_deep_learning.get():
                    from Neural_Modules.feature_integrator import FeatureIntegrator
                    feature_model_path = self.model_path.get()
                    try:
                        feature_extractor = FeatureIntegrator(model_path=feature_model_path)
                        print(f"加载深度学习模型成功: {feature_model_path}")
                    except Exception as e:
                        error_msg = str(e)
                        def show_dl_error():
                            messagebox.showerror("错误", f"加载深度学习模型失败: {error_msg}")
                        self.root.after(0, show_dl_error)
                        return
                
                # 更新进度
                self.root.after(0, lambda: self.current_progress.set(40))
                
                # 执行简化
                simplified_meshes = lod_generator.generate_lods(
                    cleaned_mesh,
                    target_faces_list=lod_levels,
                    feature_aware=self.use_deep_learning.get(),
                    use_open3d=self.use_open3d.get(),
                    feature_extractor=feature_extractor
                )
                
                # 更新进度
                self.root.after(0, lambda: self.current_progress.set(60))
                
                # 保存结果
                output_dir = self.output_model_path.get()
                if not output_dir:
                    output_dir = os.path.dirname(input_model_path)
                
                # 从输入文件名中提取基本名称
                base_name = os.path.splitext(os.path.basename(input_model_path))[0]
                
                # 保存每个LOD级别
                for i, (level, lod_mesh) in enumerate(zip(lod_levels, simplified_meshes)):
                    output_filename = f"{base_name}_lod_{level}.obj"
                    output_path = os.path.join(output_dir, output_filename)
                    o3d.io.write_triangle_mesh(output_path, lod_mesh)
                    
                    # 更新进度
                    progress = 60 + (i + 1) / len(lod_levels) * 40
                    self.root.after(0, lambda p=progress: self.current_progress.set(p))
                
                # 显示第一个LOD级别
                if simplified_meshes:
                    self.preview.show_simplified(simplified_meshes[0])
                
                # 完成后立即更新状态
                self.root.after(0, lambda: self.current_progress.set(100))
                self.root.after(0, lambda: self.status_message.set("多级LOD生成完成"))
                
                # 构建成功消息
                success_message = f"多级LOD生成完成，共生成 {len(lod_levels)} 个LOD级别"
                success_message += f"\n保存路径: {output_dir}"
                
                def show_success():
                    messagebox.showinfo("成功", success_message)
                self.root.after(0, show_success)
                
            except Exception as e:
                error_msg = str(e)
                def show_error():
                    self.status_message.set("生成多级LOD失败")
                    messagebox.showerror("错误", f"生成多级LOD失败: {error_msg}")
                self.root.after(0, show_error)
        
        # 启动线程
        thread = threading.Thread(target=lod_thread)
        thread.daemon = True
        thread.start()
    
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
    
    def clean_model(self):
        model_path = self.input_model_path.get()
        if not model_path:
            messagebox.showerror("错误", "请选择输入模型")
            return
        
        try:
            import open3d as o3d
            
            self.status_message.set("清理模型中...")
            self.root.update()
            
            # 加载模型
            mesh = o3d.io.read_triangle_mesh(model_path)
            
            # 清理模型
            cleaned_mesh = clean_mesh(mesh)
            
            # 保存清理后的模型
            output_dir = self.output_model_path.get()
            if not output_dir:
                output_dir = os.path.dirname(model_path)
            
            base_name = os.path.splitext(os.path.basename(model_path))[0]
            output_filename = f"{base_name}_cleaned.obj"
            output_path = os.path.join(output_dir, output_filename)
            
            o3d.io.write_triangle_mesh(output_path, cleaned_mesh)
            
            self.status_message.set("模型清理完成")
            messagebox.showinfo("成功", f"模型清理完成并保存为: {output_path}")
            
        except Exception as e:
            self.status_message.set("模型清理失败")
            messagebox.showerror("错误", f"模型清理失败: {str(e)}")
    
    def evaluate_model(self):
        if not hasattr(self, 'simplified_mesh'):
            messagebox.showerror("错误", "请先执行减面")
            return
        
        try:
            import open3d as o3d
            
            # 加载原始模型
            original_mesh = o3d.io.read_triangle_mesh(self.input_model_path.get())
            
            # 生成评估报告
            self.generate_evaluation_report(original_mesh, self.simplified_mesh)
            
            messagebox.showinfo("成功", "模型评估完成，报告已生成")
            
        except Exception as e:
            messagebox.showerror("错误", f"模型评估失败: {str(e)}")
    
    def generate_evaluation_report(self, original_mesh, simplified_mesh):
        """生成模型评估报告"""
        try:
            import open3d as o3d
            import numpy as np
            
            # 计算原始模型和简化模型的顶点数和面数
            original_vertices = np.asarray(original_mesh.vertices)
            original_triangles = np.asarray(original_mesh.triangles)
            simplified_vertices = np.asarray(simplified_mesh.vertices)
            simplified_triangles = np.asarray(simplified_mesh.triangles)
            
            # 计算简化率
            vertex_reduction = (1 - len(simplified_vertices) / len(original_vertices)) * 100
            face_reduction = (1 - len(simplified_triangles) / len(original_triangles)) * 100
            
            # 计算Hausdorff距离（近似）
            # 注意：完整的Hausdorff距离计算可能很耗时，这里使用近似方法
            hausdorff_distance = 0.0
            try:
                # 计算原始模型顶点到简化模型的最小距离
                dists = []
                for v in original_vertices[:1000]:  # 只取前1000个顶点以加快计算
                    min_dist = float('inf')
                    for sv in simplified_vertices:
                        dist = np.linalg.norm(v - sv)
                        if dist < min_dist:
                            min_dist = dist
                    dists.append(min_dist)
                hausdorff_distance = np.max(dists) if dists else 0.0
            except Exception as e:
                hausdorff_distance = -1  # 计算失败
            
            # 生成报告
            report = f"模型评估报告\n"
            report += f"=====================================\n"
            report += f"原始模型: {len(original_vertices)} 顶点, {len(original_triangles)} 面\n"
            report += f"简化模型: {len(simplified_vertices)} 顶点, {len(simplified_triangles)} 面\n"
            report += f"顶点减少率: {vertex_reduction:.2f}%\n"
            report += f"面数减少率: {face_reduction:.2f}%\n"
            if hausdorff_distance >= 0:
                report += f"近似Hausdorff距离: {hausdorff_distance:.4f}\n"
            else:
                report += f"Hausdorff距离计算失败\n"
            report += f"=====================================\n"
            
            # 显示报告
            if hasattr(self, 'eval_text') and self.eval_text:
                self.eval_text.delete(1.0, tk.END)
                self.eval_text.insert(tk.END, report)
            
        except Exception as e:
            if hasattr(self, 'eval_text') and self.eval_text:
                self.eval_text.delete(1.0, tk.END)
                self.eval_text.insert(tk.END, f"生成评估报告失败: {str(e)}")
    
    def show_about(self):
        messagebox.showinfo("关于", "网格简化工具 v1.0\n\n基于深度学习的特征感知网格简化工具")
    
    def show_help(self):
        help_text = "使用说明:\n\n"
        help_text += "1. 点击'浏览'按钮选择输入模型文件\n"
        help_text += "2. 选择输出目录\n"
        help_text += "3. 设置目标面数\n"
        help_text += "4. 选择是否使用Open3D加速或深度学习特征\n"
        help_text += "5. 点击'加载模型'按钮加载模型\n"
        help_text += "6. 点击'执行减面'按钮执行网格简化\n"
        help_text += "7. 点击'生成多级LOD'按钮生成多个LOD级别\n"
        help_text += "8. 点击'保存结果'按钮保存简化后的模型\n"
        messagebox.showinfo("使用说明", help_text)
    
    def run(self):
        self.root.mainloop()