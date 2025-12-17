import numpy as np
import open3d as o3d


def detect_creases(mesh, dihedral_angle_threshold=45.0):
    """检测网格中的折痕边（基于二面角）"""
    if not mesh.has_triangles() or len(mesh.triangles) < 2:
        return set()
    
    # 转换为numpy数组以便高效计算
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    
    # 构建边到三角形的映射
    edge_to_triangles = {}
    
    for tri_idx, tri in enumerate(triangles):
        # 获取三角形的三条边（排序后作为键）
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]]))
        ]
        
        # 记录每个边所属的三角形
        for edge in edges:
            if edge not in edge_to_triangles:
                edge_to_triangles[edge] = []
            edge_to_triangles[edge].append(tri_idx)
    
    crease_edges = set()
    
    # 计算每个边的二面角
    for edge, tri_indices in edge_to_triangles.items():
        # 只处理共享两个三角形的边
        if len(tri_indices) != 2:
            continue
        
        # 获取两个三角形
        tri1 = triangles[tri_indices[0]]
        tri2 = triangles[tri_indices[1]]
        
        # 计算两个三角形的法线
        normal1 = compute_triangle_normal(vertices[tri1[0]], vertices[tri1[1]], vertices[tri1[2]])
        normal2 = compute_triangle_normal(vertices[tri2[0]], vertices[tri2[1]], vertices[tri2[2]])
        
        # 计算二面角（弧度）
        dot_product = np.dot(normal1, normal2)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # 避免浮点误差
        dihedral_angle = np.arccos(dot_product)
        
        # 转换为度数
        dihedral_angle_deg = np.degrees(dihedral_angle)
        
        # 如果二面角超过阈值，标记为折痕边
        if dihedral_angle_deg > dihedral_angle_threshold:
            crease_edges.add(edge)
    
    return crease_edges

def compute_triangle_normal(v1, v2, v3):
    """计算三角形的法线向量"""
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal = np.cross(edge1, edge2)
    normal_length = np.linalg.norm(normal)
    
    if normal_length > 1e-8:
        return normal / normal_length
    else:
        return np.array([0, 0, 0])

def mark_creases(mesh, crease_edges):
    """在网格上标记折痕边"""
    # 创建线集来可视化折痕边
    lines = []
    colors = []
    
    for edge in crease_edges:
        lines.append([edge[0], edge[1]])
        colors.append([1.0, 0.0, 0.0])  # 红色表示折痕边
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set