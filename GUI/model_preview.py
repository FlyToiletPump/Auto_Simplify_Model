#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型预览功能
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import open3d as o3d
import threading

class ModelPreview(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        # 创建分割窗口
        self.paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # 原始模型预览
        self.original_frame = tk.Frame(self.paned_window, bg="lightgray")
        self.paned_window.add(self.original_frame, weight=1)
        tk.Label(self.original_frame, text="原始模型", bg="lightgray").pack()
        
        # 简化模型预览
        self.simplified_frame = tk.Frame(self.paned_window, bg="lightgray")
        self.paned_window.add(self.simplified_frame, weight=1)
        tk.Label(self.simplified_frame, text="简化模型", bg="lightgray").pack()
        
        # 模型信息
        self.info_frame = tk.Frame(self, bg="white", relief=tk.SUNKEN, borderwidth=1)
        self.info_frame.pack(fill=tk.X, pady=5)
        
        self.original_info = tk.StringVar(value="原始模型: 0 顶点, 0 面")
        self.simplified_info = tk.StringVar(value="简化模型: 0 顶点, 0 面")
        
        tk.Label(self.info_frame, textvariable=self.original_info).pack(side=tk.LEFT, padx=10)
        tk.Label(self.info_frame, textvariable=self.simplified_info).pack(side=tk.RIGHT, padx=10)
        
        # 预览窗口引用
        self.original_visualizer = None
        self.simplified_visualizer = None
    
    def load_model(self, model_path):
        """加载模型并显示信息"""
        try:
            mesh = o3d.io.read_triangle_mesh(model_path)
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            self.original_info.set(f"原始模型: {len(vertices)} 顶点, {len(triangles)} 面")
            
            # 显示3D模型
            self.show_3d_model(mesh, "原始模型", "original")
            
        except Exception as e:
            raise e
    
    def set_original_info(self, vertices_count, triangles_count):
        """直接设置原始模型信息，不重新加载模型"""
        self.original_info.set(f"原始模型: {vertices_count} 顶点, {triangles_count} 面")
    
    def set_simplified_info(self, vertices_count, triangles_count):
        """直接设置简化模型信息"""
        self.simplified_info.set(f"简化模型: {vertices_count} 顶点, {triangles_count} 面")
    
    def show_3d_model_async(self, mesh, window_name, model_type):
        """异步显示3D模型，不阻塞UI"""
        thread = threading.Thread(target=self._show_3d_model_thread, args=(mesh, window_name, model_type))
        thread.daemon = True
        thread.start()
    
    def _show_3d_model_thread(self, mesh, window_name, model_type):
        """后台线程显示3D模型"""
        try:
            import time
            time.sleep(0.5)  # 等待UI稳定
            
            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window(window_name=window_name, width=800, height=600, visible=True)
            
            visualizer.add_geometry(mesh)
            visualizer.get_render_option().background_color = [0.8, 0.8, 0.8]
            visualizer.get_render_option().mesh_show_wireframe = False
            visualizer.get_render_option().mesh_show_back_face = True
            visualizer.reset_view_point(True)
            
            while visualizer.poll_events():
                visualizer.update_renderer()
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Visualizer error: {e}")
        finally:
            try:
                if visualizer:
                    visualizer.destroy_window()
            except:
                pass
    
    def show_simplified(self, simplified_mesh):
        """显示简化后的模型信息"""
        try:
            vertices = np.asarray(simplified_mesh.vertices)
            triangles = np.asarray(simplified_mesh.triangles)
            
            self.simplified_info.set(f"简化模型: {len(vertices)} 顶点, {len(triangles)} 面")
            
            # 使用异步方式显示3D模型，不阻塞UI
            self.show_3d_model_async(simplified_mesh, "简化模型", "simplified")
            
        except Exception as e:
            raise e
    
    def show_3d_model(self, mesh, window_name, model_type):
        """显示3D模型"""
        # 关闭之前的窗口
        if model_type == "original" and hasattr(self, 'original_visualizer') and self.original_visualizer:
            try:
                self.original_visualizer.destroy_window()
                self.original_visualizer = None
            except:
                pass
        elif model_type == "simplified" and hasattr(self, 'simplified_visualizer') and self.simplified_visualizer:
            try:
                self.simplified_visualizer.destroy_window()
                self.simplified_visualizer = None
            except:
                pass
        
        # 在后台线程中运行可视化器，避免阻塞UI
        def run_visualizer():
            try:
                import time
                time.sleep(0.1)  # 等待UI更新完成
                
                visualizer = o3d.visualization.Visualizer()
                visualizer.create_window(window_name=window_name, width=800, height=600, visible=True)
                
                visualizer.add_geometry(mesh)
                visualizer.get_render_option().background_color = [0.8, 0.8, 0.8]
                visualizer.get_render_option().mesh_show_wireframe = False
                visualizer.get_render_option().mesh_show_back_face = True
                visualizer.reset_view_point(True)
                
                while visualizer.poll_events():
                    visualizer.update_renderer()
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"Visualizer error: {e}")
            finally:
                try:
                    if visualizer:
                        visualizer.destroy_window()
                except:
                    pass
        
        thread = threading.Thread(target=run_visualizer)
        thread.daemon = True
        thread.start()