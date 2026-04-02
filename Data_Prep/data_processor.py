import os
import numpy as np
import open3d as o3d
import json
import h5py

class DataProcessor:
    """数据处理器，用于生成训练数据"""
    
    def __init__(self, data_dir=None):
        """
        参数:
            data_dir: 数据目录路径
        """
        # 如果没有提供数据目录，使用项目根目录下的Data文件夹
        if data_dir is None:
            # 获取当前文件的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 项目根目录是当前目录的上一级
            project_root = os.path.abspath(os.path.join(current_dir, ".."))
            self.data_dir = os.path.join(project_root, "Data")
        else:
            self.data_dir = data_dir
        
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.annotations_dir = os.path.join(self.data_dir, "annotations")
        
        # 确保目录存在
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
    
    def load_mesh(self, mesh_path):
        """加载网格模型"""
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()
        return mesh
    
    def extract_vertex_features(self, mesh):
        """提取顶点特征"""
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        features = np.concatenate([vertices, normals], axis=1)
        return features
    
    def compute_vertex_importance(self, mesh):
        """计算顶点重要性
        基于几何特征自动生成标注
        """
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        triangles = np.asarray(mesh.triangles)
        
        # 计算顶点的度
        vertex_degree = np.zeros(len(vertices), dtype=int)
        for tri in triangles:
            vertex_degree[tri[0]] += 1
            vertex_degree[tri[1]] += 1
            vertex_degree[tri[2]] += 1
        
        # 计算顶点的曲率（基于法线变化）
        vertex_curvature = np.zeros(len(vertices))
        for i in range(len(vertices)):
            # 找到与该顶点相连的所有三角形
            adjacent_tris = []
            for tri in triangles:
                if i in tri:
                    adjacent_tris.append(tri)
            
            # 计算相邻三角形法线的变化
            if len(adjacent_tris) > 1:
                normals_list = []
                for tri in adjacent_tris:
                    tri_normal = np.asarray(mesh.triangle_normals)[tri.tolist().index(i)]
                    normals_list.append(tri_normal)
                
                # 计算法线之间的角度差异
                angle_diff = 0
                for j in range(len(normals_list) - 1):
                    dot = np.dot(normals_list[j], normals_list[j+1])
                    angle_diff += np.arccos(np.clip(dot, -1, 1))
                vertex_curvature[i] = angle_diff / (len(normals_list) - 1)
        
        # 计算顶点的边界性
        vertex_boundary = np.zeros(len(vertices))
        # 构建边集合
        edges = set()
        edge_counts = {}
        for tri in triangles:
            edges.add(tuple(sorted((tri[0], tri[1]))))
            edges.add(tuple(sorted((tri[1], tri[2]))))
            edges.add(tuple(sorted((tri[2], tri[0]))))
        
        # 统计每条边的出现次数
        for edge in edges:
            if edge in edge_counts:
                edge_counts[edge] += 1
            else:
                edge_counts[edge] = 1
        
        # 边界边是只出现一次的边
        boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
        
        # 标记边界顶点
        for edge in boundary_edges:
            vertex_boundary[edge[0]] = 1
            vertex_boundary[edge[1]] = 1
        
        # 综合计算重要性
        # 归一化各项特征
        degree_norm = (vertex_degree - vertex_degree.min()) / (vertex_degree.max() - vertex_degree.min() + 1e-8)
        curvature_norm = (vertex_curvature - vertex_curvature.min()) / (vertex_curvature.max() - vertex_curvature.min() + 1e-8)
        
        # 权重
        w_degree = 0.3
        w_curvature = 0.5
        w_boundary = 0.2
        
        importance = w_degree * degree_norm + w_curvature * curvature_norm + w_boundary * vertex_boundary
        
        # 归一化到[0, 1]
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        
        return importance
    
    def process_mesh(self, mesh_path, output_name):
        """处理单个网格并生成训练数据"""
        try:
            # 加载网格
            mesh = self.load_mesh(mesh_path)
            
            # 提取特征
            features = self.extract_vertex_features(mesh)
            
            # 计算重要性
            importance = self.compute_vertex_importance(mesh)
            
            # 保存处理后的数据
            output_file = os.path.join(self.processed_dir, f"{output_name}.h5")
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('features', data=features)
                f.create_dataset('importance', data=importance)
                f.create_dataset('vertices', data=np.asarray(mesh.vertices))
                f.create_dataset('triangles', data=np.asarray(mesh.triangles))
            
            print(f"处理完成: {output_name}")
            return True
        except Exception as e:
            print(f"处理失败 {output_name}: {e}")
            return False
    
    def process_directory(self, input_dir, output_prefix):
        """处理目录中的所有网格文件"""
        mesh_files = []
        for ext in ['.obj', '.ply', '.stl']:
            mesh_files.extend([f for f in os.listdir(input_dir) if f.endswith(ext)])
        
        print(f"找到 {len(mesh_files)} 个网格文件")
        
        success_count = 0
        for i, mesh_file in enumerate(mesh_files):
            mesh_path = os.path.join(input_dir, mesh_file)
            output_name = f"{output_prefix}_{i}"
            if self.process_mesh(mesh_path, output_name):
                success_count += 1
        
        print(f"处理完成: {success_count}/{len(mesh_files)} 成功")
        return success_count
    
    def generate_synthetic_data(self, num_samples=20000, output_file="synthetic_data.h5"):
        """生成合成训练数据"""
        # 生成随机顶点特征
        features = np.random.rand(num_samples, 6)  # 3D坐标 + 3D法线
        
        # 生成合成重要性标签
        # 基于多个特征模式生成标签，增加数据多样性
        # 1. 法线的z分量
        importance_z = features[:, 5]
        # 2. 坐标的x分量
        importance_x = features[:, 0]
        # 3. 坐标的y分量
        importance_y = features[:, 1]
        # 4. 法线的长度（虽然法线通常是单位向量，但这里模拟一些变化）
        normal_length = np.linalg.norm(features[:, 3:6], axis=1)
        importance_normal = (normal_length - normal_length.min()) / (normal_length.max() - normal_length.min() + 1e-8)
        
        # 综合多个特征计算重要性
        importance = 0.3 * importance_z + 0.2 * importance_x + 0.2 * importance_y + 0.3 * importance_normal
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        
        # 添加一些噪声
        noise = np.random.rand(num_samples) * 0.1
        importance = np.clip(importance + noise, 0, 1)
        
        # 保存数据
        output_path = os.path.join(self.processed_dir, output_file)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('features', data=features)
            f.create_dataset('importance', data=importance)
        
        print(f"合成数据生成完成: {output_file}")
        return output_path

if __name__ == "__main__":
    # 示例用法
    processor = DataProcessor()
    
    # 生成合成数据
    processor.generate_synthetic_data()
    
    # 如果有真实模型，可以使用以下代码处理
    # processor.process_directory("path/to/meshes", "mesh")
