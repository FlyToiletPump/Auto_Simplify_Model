# auto_simplify_mvp.py
import argparse
import json
import os

import numpy as np
import open3d as o3d


def load_mesh(path):
    """安全加载网格，自动处理常见问题"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) == 0:
        raise ValueError("Loaded mesh has no vertices.")
    if len(mesh.triangles) == 0:
        raise ValueError("Loaded mesh has no faces (is it a point cloud?).")

    # 确保有法线（简化需要）
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    return mesh


def clean_and_normalize(mesh):
    """清理网格并归一化到单位球内，提升简化稳定性"""
    # 移除重复和退化元素
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    # 归一化到 [-1, 1] 范围（可选，但推荐）
    center = mesh.get_center()
    mesh.translate(-center)
    extent = np.max(np.linalg.norm(np.asarray(mesh.vertices), axis=1))
    if extent > 0:
        mesh.scale(0.9 / extent, center=[0, 0, 0])

    mesh.compute_vertex_normals()
    return mesh


def simplify_mesh(mesh, target_faces):
    """安全简化，带面数检查"""
    original_faces = len(mesh.triangles)
    print(f"Original faces: {original_faces}")

    if original_faces <= target_faces:
        print("No simplification needed.")
        return mesh

    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)

    if len(simplified.triangles) == 0:
        raise RuntimeError("Simplification failed: result is empty.")

    simplified.compute_vertex_normals()
    print(f"Simplified faces: {len(simplified.triangles)}")
    return simplified


def save_mesh_safe(mesh, output_path):
    """强制保存为 ASCII PLY（最兼容），或 OBJ"""
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".ply":
        # 关键：write_ascii=True 避免二进制魔数问题
        o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=True)
    elif ext == ".obj":
        o3d.io.write_triangle_mesh(output_path, mesh)
    else:
        # 默认转为 PLY
        output_path = output_path.replace(ext, ".ply")
        o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=True)

    # 验证是否能重读
    test_load = o3d.io.read_triangle_mesh(output_path)
    if len(test_load.vertices) == 0:
        raise IOError(f"Failed to save valid mesh to {output_path}")

    print(f"Saved and verified: {output_path}")
    return output_path


def evaluate_simplification(original_mesh, simplified_mesh, sample_points=10000):
    """
    兼容旧版 Open3D 的简化质量评估（仅使用点云采样 + 点云距离）
    """
    # 从原始网格和简化网格分别采样点云
    pcd_orig = original_mesh.sample_points_uniformly(number_of_points=sample_points)
    pcd_simp = simplified_mesh.sample_points_uniformly(number_of_points=sample_points)

    # 方向1: 原始点云 → 简化点云
    dist_orig_to_simp = np.asarray(pcd_orig.compute_point_cloud_distance(pcd_simp))

    # 方向2: 简化点云 → 原始点云
    dist_simp_to_orig = np.asarray(pcd_simp.compute_point_cloud_distance(pcd_orig))

    # 双向 Hausdorff 距离（近似）
    hausdorff = max(np.max(dist_orig_to_simp), np.max(dist_simp_to_orig))
    # RMS 误差（通常用原始→简化）
    rms = np.sqrt(np.mean(np.square(dist_orig_to_simp)))

    return {"hausdorff": float(hausdorff), "rms": float(rms)}


def main():
    parser = argparse.ArgumentParser(description="Robust 3D mesh simplification with quality evaluation.")
    parser.add_argument("--input", required=True, help="Input mesh file (.obj, .ply, .stl, .glb, etc.)")
    parser.add_argument("--output", default="output.ply", help="Output file (recommended: .ply or .obj)")
    parser.add_argument("--target_faces", type=int, default=5000,
                        help="Target number of triangles after simplification")
    args = parser.parse_args()

    # 1. Load and validate
    mesh_orig = load_mesh(args.input)

    # 2. Clean and normalize (improves simplification stability)
    mesh_orig = clean_and_normalize(mesh_orig)

    # 3. Simplify
    mesh_simp = simplify_mesh(mesh_orig, args.target_faces)

    # 4. Save (guaranteed valid format)
    output_path = save_mesh_safe(mesh_simp, args.output)

    # 5. Evaluate
    metrics = evaluate_simplification(mesh_orig, mesh_simp)
    print(f"Quality metrics: Hausdorff={metrics['hausdorff']:.4f}, RMS={metrics['rms']:.4f}")

    # 6. Save metrics
    metrics_path = os.path.splitext(output_path)[0] + "_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()