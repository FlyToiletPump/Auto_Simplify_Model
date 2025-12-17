import os

import open3d as o3d


def load_mesh(path):
    """安全加载3D网格模型"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) == 0:
        raise ValueError("Loaded mesh has no vertices.")
    if len(mesh.triangles) == 0:
        raise ValueError("Loaded mesh has no faces (is it a point cloud?).")

    # 确保有法线
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    return mesh

def save_mesh(mesh, output_path, ascii_format=True):
    """安全保存3D网格模型"""
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".ply":
        o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=ascii_format)
    elif ext == ".obj":
        o3d.io.write_triangle_mesh(output_path, mesh)
    else:
        output_path = output_path.replace(ext, ".ply")
        o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=ascii_format)

    # 验证保存结果
    test_load = o3d.io.read_triangle_mesh(output_path)
    if len(test_load.vertices) == 0:
        raise IOError(f"Failed to save valid mesh to {output_path}")

    return output_path