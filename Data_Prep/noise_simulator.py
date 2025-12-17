import numpy as np
import open3d as o3d


def subdivide_mesh(mesh, subdivision_levels=1):
    """对网格进行细分，增加其复杂度"""
    for _ in range(subdivision_levels):
        mesh = mesh.subdivide_loop(number_of_iterations=1)
    mesh.compute_vertex_normals()
    return mesh

def add_geometric_noise(mesh, noise_level=0.01):
    """在网格顶点上添加随机噪声"""
    vertices = np.asarray(mesh.vertices)
    noise = np.random.normal(0, noise_level, size=vertices.shape)
    noisy_vertices = vertices + noise
    
    noisy_mesh = o3d.geometry.TriangleMesh()
    noisy_mesh.vertices = o3d.utility.Vector3dVector(noisy_vertices)
    noisy_mesh.triangles = mesh.triangles
    noisy_mesh.compute_vertex_normals()
    return noisy_mesh

def add_topological_noise(mesh, noise_ratio=0.01):
    """通过添加小三角形模拟拓扑噪声"""
    # 此函数需要更复杂的实现，这里提供简化版本
    # 实际应用中可以考虑添加小的随机三角形或修改现有边
    return mesh

def simulate_generative_output(mesh, subdivision_levels=2, noise_level=0.01):
    """模拟生成模型输出的不规则网格"""
    # 1. 细分网格增加复杂度
    mesh = subdivide_mesh(mesh, subdivision_levels)
    
    # 2. 添加几何噪声
    mesh = add_geometric_noise(mesh, noise_level)
    
    # 3. 添加拓扑噪声（可选）
    # mesh = add_topological_noise(mesh, 0.01)
    
    # 4. 确保网格有效
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    
    return mesh

def generate_datasets(original_mesh, num_samples=5, noise_levels=[0.01, 0.03, 0.05]):
    """生成用于训练的数据集"""
    dataset = []
    
    for i in range(num_samples):
        for noise_level in noise_levels:
            # 为每个样本生成不同的噪声
            noisy_mesh = simulate_generative_output(
                original_mesh, 
                subdivision_levels=2, 
                noise_level=noise_level
            )
            dataset.append({
                "id": f"sample_{i}_noise_{noise_level:.3f}",
                "mesh": noisy_mesh,
                "noise_level": noise_level
            })
    
    return dataset