import numpy as np
import open3d as o3d

orig = o3d.io.read_triangle_mesh("../bunny_10k.obj")
simp = o3d.io.read_triangle_mesh("../simplified_bunny.ply")

print(f"Original: {len(orig.triangles)} faces")
print(f"Simplified: {len(simp.triangles)} faces")

# 归一化
for m in [orig, simp]:
    c = m.get_center()
    m.translate(-c)
    s = np.max(np.linalg.norm(np.asarray(m.vertices), axis=1))
    if s > 0: m.scale(0.7 / s, center=[0,0,0])

simp.translate([2, 0, 0])
orig.paint_uniform_color([0.8, 0.8, 0.8])
simp.paint_uniform_color([1, 0.65, 0])

o3d.visualization.draw_geometries(
    [orig, simp],
    window_name=f"Original vs Simplified",
    width=1200, height=800
)

