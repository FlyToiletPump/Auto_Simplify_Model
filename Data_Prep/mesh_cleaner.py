import numpy as np
import open3d as o3d


def clean_mesh(mesh):
    """清理网格中的无效元素"""
    # 移除重复顶点和三角形
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    
    # 移除退化三角形
    mesh.remove_degenerate_triangles()
    
    # 修复非流形边
    mesh = mesh.remove_non_manifold_edges()
    
    return mesh

def normalize_mesh(mesh, scale=0.9):
    """将网格归一化到单位球内"""
    # 平移到原点
    center = mesh.get_center()
    mesh.translate(-center)
    
    # 计算最大范围
    vertices = np.asarray(mesh.vertices)
    max_extent = np.max(np.linalg.norm(vertices, axis=1))
    
    # 缩放到指定大小
    if max_extent > 0:
        mesh.scale(scale / max_extent, center=[0, 0, 0])
    
    return mesh

def remove_noise(mesh, smoothing_iterations=5, lambda_factor=0.5):
    """使用拉普拉斯平滑减少网格噪声"""
    smoothed_mesh = mesh.filter_smooth_laplacian(
        number_of_iterations=smoothing_iterations,
        lambda_factor=lambda_factor,
        filter_scope=o3d.geometry.FilterScope.All
    )
    smoothed_mesh.compute_vertex_normals()
    return smoothed_mesh

def preprocess_mesh(mesh, smoothing_iterations=0):
    """完整的网格预处理流程"""
    # 清理网格
    mesh = clean_mesh(mesh)
    
    # 归一化
    mesh = normalize_mesh(mesh)
    
    # 去噪（如果需要）
    if smoothing_iterations > 0:
        mesh = remove_noise(mesh, smoothing_iterations)
    
    mesh.compute_vertex_normals()
    return mesh