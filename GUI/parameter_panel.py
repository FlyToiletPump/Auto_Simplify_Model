#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数设置面板
"""

import tkinter as tk
from tkinter import filedialog

class ParameterPanel(tk.Frame):
    def __init__(self, parent, target_faces, use_open3d, use_deep_learning, model_path):
        super().__init__(parent, relief=tk.RAISED, borderwidth=1)
        
        self.target_faces = target_faces
        self.use_open3d = use_open3d
        self.use_deep_learning = use_deep_learning
        self.model_path = model_path
        
        self.create_widgets()
    
    def create_widgets(self):
        # 目标面数设置
        face_frame = tk.Frame(self, padx=10, pady=5)
        face_frame.pack(fill=tk.X)
        
        tk.Label(face_frame, text="目标面数:").pack(side=tk.LEFT, padx=5)
        tk.Entry(face_frame, textvariable=self.target_faces, width=10).pack(side=tk.LEFT, padx=5)
        
        # 选项设置
        option_frame = tk.Frame(self, padx=10, pady=5)
        option_frame.pack(fill=tk.X)
        
        tk.Checkbutton(option_frame, text="使用Open3D加速", variable=self.use_open3d).pack(side=tk.LEFT, padx=10)
        
        dl_frame = tk.Frame(option_frame)
        dl_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Checkbutton(dl_frame, text="使用深度学习特征", variable=self.use_deep_learning).pack(side=tk.LEFT)
        tk.Button(dl_frame, text="浏览", command=self.browse_model).pack(side=tk.LEFT, padx=5)
        tk.Entry(dl_frame, textvariable=self.model_path, width=30).pack(side=tk.LEFT, padx=5)
    
    def browse_model(self):
        filename = filedialog.askopenfilename(
            filetypes=[("模型文件", "*.pth"), ("所有文件", "*.*")]
        )
        if filename:
            self.model_path.set(filename)
