"""
Step 2: Clean both facade scans and build 2.5D grid meshes.

Pipeline per scan:
  1. Load (scan 1 raw, scan 2 from PCA-aligned npy)
  2. Depth (X) clip — keep 2nd–98th percentile; removes far background & noise
  3. Z clip — trim to actual building envelope (2nd–97th percentile)
  4. Statistical outlier removal (k-NN mean distance)
  5. 2.5D grid mesh — project onto Y-Z facade plane, use median X as surface depth
     Grid cells → vertices → triangle strip; NaN cells filled by nearest-neighbour
  6. Save mesh as .npz (vertices, faces, vertex colours)
  7. Audit plot: cleaned point cloud + mesh surface coloured by depth

Convention: X=depth, Y=along-street, Z=height
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.interpolate import NearestNDInterpolator
import os

# ── parameters ────────────────────────────────────────────────────────────────

# Per-scan grid parameters — scan B is sparser (54 pts/m² vs 300 pts/m² scan A)
GRID_RES_A  = 0.12   # metres — scan A (dense, 12 cm)
GRID_RES_B  = 0.30   # metres — scan B (sparse, 30 cm gives ~5 pts/cell)
MIN_PTS_A   = 3      # scan A can afford ≥3 pts/cell for robust median
MIN_PTS_B   = 1      # scan B: accept any point

SOR_K       = 20     # statistical outlier removal: k neighbours
SOR_THRESH  = 2.0    # remove if mean-NN-dist > global_mean + thresh*std
MIN_PTS_CELL = 3     # default (overridden per scan below)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_ply(path):
    with open(path, 'rb') as f:
        while f.readline().strip() != b'end_header':
            pass
        raw = f.read()
    n = len(raw) // 27
    data = np.frombuffer(raw[:n*27], dtype=np.dtype([
        ('x','<f8'),('y','<f8'),('z','<f8'),('r','u1'),('g','u1'),('b','u1')
    ]))
    xyz = np.stack([data['x'], data['y'], data['z']], axis=1).copy()
    rgb = np.stack([data['r'], data['g'], data['b']], axis=1) / 255.0
    return xyz, rgb


def pca_align(xyz):
    """Same alignment used in align_and_audit.py — reproduced here for self-containment."""
    centroid = xyz.mean(0)
    centered = xyz - centroid
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc_street, pc_height, pc_depth = Vt[0], Vt[1], Vt[2]
    R = np.column_stack([pc_depth, pc_street, pc_height])
    aligned = centered @ R
    for col in range(3):
        if np.median(aligned[:, col]) < 0:
            R[:, col] *= -1; aligned[:, col] *= -1
    z_med = np.median(aligned[:, 2])
    if aligned[aligned[:, 2] > z_med, 2].mean() < aligned[aligned[:, 2] < z_med, 2].mean():
        R[:, 2] *= -1; aligned[:, 2] *= -1
    return aligned, centroid, R


def clip_percentile(xyz, rgb, axis, lo_pct, hi_pct):
    lo = np.percentile(xyz[:, axis], lo_pct)
    hi = np.percentile(xyz[:, axis], hi_pct)
    mask = (xyz[:, axis] >= lo) & (xyz[:, axis] <= hi)
    return xyz[mask], rgb[mask], (lo, hi)


def stat_outlier_removal(xyz, rgb, k=20, thresh=2.0):
    tree = KDTree(xyz)
    dists, _ = tree.query(xyz, k=k+1)
    mean_d = dists[:, 1:].mean(axis=1)
    cutoff = mean_d.mean() + thresh * mean_d.std()
    mask = mean_d < cutoff
    print(f"    SOR removed {(~mask).sum():,} / {len(xyz):,} points "
          f"({100*(~mask).mean():.1f}%)")
    return xyz[mask], rgb[mask]


def build_2d5_mesh(xyz, rgb, res=0.12, min_pts=3):
    """
    Project facade points onto the Y-Z plane and build a 2.5D height-map mesh.

    Grid is defined over (Y, Z) with cell size `res`.
    For each cell, median X (depth) is stored as the surface displacement.
    Empty cells are filled by nearest-neighbour interpolation.

    Returns
    -------
    verts  : (V, 3) float64 — 3-D vertex positions
    faces  : (F, 3) int32   — triangle indices
    vcolors: (V, 3) float64 — per-vertex RGB (median of cell points)
    grid_info: dict with grid metadata
    """
    Y, Z, X = xyz[:, 1], xyz[:, 2], xyz[:, 0]

    y_min, y_max = Y.min(), Y.max()
    z_min, z_max = Z.min(), Z.max()

    ny = int(np.ceil((y_max - y_min) / res)) + 1
    nz = int(np.ceil((z_max - z_min) / res)) + 1
    print(f"    Grid: {ny} × {nz} = {ny*nz:,} cells  ({res*100:.0f} cm resolution)")

    # Map each point to its grid cell
    iy = np.floor((Y - y_min) / res).astype(int)
    iz = np.floor((Z - z_min) / res).astype(int)

    # Per-cell: median X and median RGB
    depth_grid  = np.full((ny, nz), np.nan)
    color_grid  = np.full((ny, nz, 3), np.nan)
    count_grid  = np.zeros((ny, nz), dtype=int)

    # Accumulate (use dict-of-lists then batch median)
    from collections import defaultdict
    cell_x   = defaultdict(list)
    cell_rgb = defaultdict(list)

    for idx in range(len(xyz)):
        key = (iy[idx], iz[idx])
        cell_x[key].append(X[idx])
        cell_rgb[key].append(rgb[idx])

    filled = 0
    for (ci, cj), xs in cell_x.items():
        if len(xs) >= min_pts:
            depth_grid[ci, cj] = np.median(xs)
            color_grid[ci, cj] = np.median(cell_rgb[(ci, cj)], axis=0)
            count_grid[ci, cj] = len(xs)
            filled += 1

    coverage = filled / (ny * nz) * 100
    print(f"    Cells with data: {filled:,} / {ny*nz:,} ({coverage:.1f}% coverage)")

    # Nearest-neighbour fill for empty cells (so mesh has no holes)
    known_mask = ~np.isnan(depth_grid)
    if known_mask.sum() < 4:
        raise ValueError("Too few occupied cells to build mesh")

    grid_ii, grid_jj = np.mgrid[0:ny, 0:nz]
    known_ij  = np.column_stack([grid_ii[known_mask], grid_jj[known_mask]])
    known_dep = depth_grid[known_mask]
    known_col = color_grid[known_mask]

    interp_dep = NearestNDInterpolator(known_ij, known_dep)
    interp_col = NearestNDInterpolator(known_ij, known_col)

    all_ij = np.column_stack([grid_ii.ravel(), grid_jj.ravel()])
    depth_flat = interp_dep(all_ij)
    color_flat = interp_col(all_ij)
    depth_grid = depth_flat.reshape(ny, nz)
    color_grid = color_flat.reshape(ny, nz, 3)

    # Build vertex array  shape (ny*nz, 3)
    y_coords = y_min + grid_ii * res   # (ny, nz)
    z_coords = z_min + grid_jj * res
    x_coords = depth_grid              # surface depth

    verts   = np.column_stack([x_coords.ravel(), y_coords.ravel(), z_coords.ravel()])
    vcolors = color_grid.reshape(-1, 3)

    # Build faces: each (ny-1)×(nz-1) quad → 2 triangles
    # vertex index at (i,j) = i*nz + j
    i_idx = np.arange(ny - 1)
    j_idx = np.arange(nz - 1)
    II, JJ = np.meshgrid(i_idx, j_idx, indexing='ij')
    II = II.ravel(); JJ = JJ.ravel()

    v00 = II * nz + JJ
    v10 = (II + 1) * nz + JJ
    v01 = II * nz + (JJ + 1)
    v11 = (II + 1) * nz + (JJ + 1)

    tri1 = np.column_stack([v00, v10, v11])
    tri2 = np.column_stack([v00, v11, v01])
    faces = np.vstack([tri1, tri2]).astype(np.int32)

    print(f"    Mesh: {len(verts):,} vertices, {len(faces):,} faces")
    return verts, faces, vcolors, {'ny': ny, 'nz': nz, 'res': res,
                                    'y_min': y_min, 'z_min': z_min,
                                    'coverage': coverage}


def process_scan(label, xyz, rgb, x_clip=(2, 98), z_clip=(2, 97),
                 grid_res=0.12, min_pts=3):
    print(f"\n{'═'*55}")
    print(f"  Processing: {label}  ({len(xyz):,} points)")
    print(f"{'═'*55}")

    # 1. X (depth) clip
    xyz, rgb, x_range = clip_percentile(xyz, rgb, 0, *x_clip)
    print(f"  X clip [{x_range[0]:.2f}, {x_range[1]:.2f}] m  → {len(xyz):,} pts")

    # 2. Z (height) clip
    xyz, rgb, z_range = clip_percentile(xyz, rgb, 2, *z_clip)
    print(f"  Z clip [{z_range[0]:.2f}, {z_range[1]:.2f}] m  → {len(xyz):,} pts")

    # 3. Statistical outlier removal
    print(f"  Statistical outlier removal (k={SOR_K}, thresh={SOR_THRESH}σ)…")
    xyz, rgb = stat_outlier_removal(xyz, rgb, k=SOR_K, thresh=SOR_THRESH)

    print(f"  Building 2.5D mesh (res={grid_res*100:.0f} cm, min_pts={min_pts})…")
    verts, faces, vcolors, grid_info = build_2d5_mesh(xyz, rgb, res=grid_res, min_pts=min_pts)

    return xyz, rgb, verts, faces, vcolors, grid_info


# ── load scans ────────────────────────────────────────────────────────────────

print("Loading 3_21_2026.ply …")
xyz_a, rgb_a = load_ply("3_21_2026.ply")
# Floor-zero scan A
z_floor_a = np.percentile(xyz_a[:, 2], 2)
xyz_a[:, 2] -= z_floor_a

print("Loading 120thamst.ply (applying PCA alignment) …")
xyz_b_raw, rgb_b = load_ply("120thamst.ply")
xyz_b, _, _ = pca_align(xyz_b_raw)
z_floor_b = np.percentile(xyz_b[:, 2], 2)
xyz_b[:, 2] -= z_floor_b

# ── process each scan ─────────────────────────────────────────────────────────

xyz_a_cl, rgb_a_cl, verts_a, faces_a, vcol_a, ginfo_a = process_scan(
    "3_21_2026.ply", xyz_a.copy(), rgb_a.copy(),
    grid_res=GRID_RES_A, min_pts=MIN_PTS_A)

xyz_b_cl, rgb_b_cl, verts_b, faces_b, vcol_b, ginfo_b = process_scan(
    "120thamst.ply (aligned)", xyz_b.copy(), rgb_b.copy(),
    grid_res=GRID_RES_B, min_pts=MIN_PTS_B)

# ── save meshes ───────────────────────────────────────────────────────────────

np.savez("mesh_a.npz", verts=verts_a, faces=faces_a, vcolors=vcol_a)
np.savez("mesh_b.npz", verts=verts_b, faces=faces_b, vcolors=vcol_b)
print("\nSaved mesh_a.npz and mesh_b.npz")

# ── audit plot ────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 14))
fig.suptitle("Cleaned point clouds + 2.5D mesh surfaces", fontsize=12)

pairs = [
    ("3_21_2026.ply",         xyz_a_cl, rgb_a_cl, verts_a, ginfo_a, 0),
    ("120thamst.ply (aligned)",xyz_b_cl, rgb_b_cl, verts_b, ginfo_b, 1),
]

for label, xyz, rgb, verts, ginfo, row_offset in pairs:
    base = row_offset * 3

    # --- front view: Y vs Z coloured by RGB ---
    ax1 = fig.add_subplot(2, 3, base + 1)
    idx = np.random.choice(len(xyz), min(len(xyz), 30_000), replace=False)
    ax1.scatter(xyz[idx, 1], xyz[idx, 2], c=rgb[idx], s=0.2, alpha=0.6)
    ax1.set_xlabel("Y — street (m)"); ax1.set_ylabel("Z — height (m)")
    ax1.set_title(f"{label}\nFront view (cleaned cloud)")
    ax1.set_aspect('equal')

    # --- side profile: X vs Z ---
    ax2 = fig.add_subplot(2, 3, base + 2)
    ax2.scatter(xyz[idx, 0], xyz[idx, 2], c=rgb[idx], s=0.2, alpha=0.6)
    ax2.set_xlabel("X — depth (m)"); ax2.set_ylabel("Z — height (m)")
    ax2.set_title("Side profile (depth vs height)")
    ax2.set_aspect('equal')

    # --- mesh: depth heatmap over Y-Z ---
    ax3 = fig.add_subplot(2, 3, base + 3)
    ny, nz = ginfo['ny'], ginfo['nz']
    res = ginfo['res']
    depth_img = verts[:, 0].reshape(ny, nz)   # X = depth
    extent = [ginfo['y_min'], ginfo['y_min'] + ny*res,
              ginfo['z_min'], ginfo['z_min'] + nz*res]
    im = ax3.imshow(depth_img.T, origin='lower', aspect='equal',
                    extent=extent, cmap='viridis')
    plt.colorbar(im, ax=ax3, label='depth X (m)', shrink=0.7)
    ax3.set_xlabel("Y — street (m)"); ax3.set_ylabel("Z — height (m)")
    ax3.set_title(f"Mesh depth heatmap\n{ny}×{nz} grid, {res*100:.0f} cm res  "
                  f"({ginfo['coverage']:.0f}% native coverage)")

plt.tight_layout()
plt.savefig("clean_mesh_audit.png", dpi=150)
plt.close()
print("Saved clean_mesh_audit.png")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "─"*55)
print("SUMMARY")
print("─"*55)
for label, v, f, g in [("Scan A", verts_a, faces_a, ginfo_a),
                        ("Scan B", verts_b, faces_b, ginfo_b)]:
    print(f"  {label}: {len(v):,} verts  {len(f):,} faces  "
          f"grid {g['ny']}×{g['nz']}  coverage {g['coverage']:.0f}%")
print("Ready for Laplace-Beltrami eigendecomposition.")
