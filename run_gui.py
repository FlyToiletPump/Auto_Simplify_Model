#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网格简化GUI入口脚本
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from GUI.main_window import MainWindow

if __name__ == "__main__":
    app = MainWindow()
    app.run()
