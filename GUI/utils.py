#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具函数
"""

import os
import sys

def get_project_root():
    """获取项目根目录"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, ".."))

def add_project_path():
    """添加项目路径到Python路径"""
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
