"""
PCA alignment of 120thamst.ply to match 3_21_2026.ply coordinate convention,
then side-by-side audit plot confirming both scans are in the same frame.

Coordinate convention (from 3_21_2026.ply):
  X  — depth from scanner (~2 m span, thin dimension)
  Y  — along street      (~52 m span, longest dimension)
  Z  — height            (~8.6 m span, vertical dimension)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import struct, os

# ── helpers ─────────────────────────────────────────────────────────────────

def load_ply(path):
    """Read binary-little-endian PLY with XYZ float64 + RGB uint8 (27 bytes/pt)."""
    with open(path, 'rb') as f:
        lines = []
        while True:
            line = f.readline().strip()
            lines.append(line)
            if line == b'end_header':
                break
        raw = f.read()
    n = len(raw) // 27
    data = np.frombuffer(raw[:n * 27], dtype=np.dtype([
        ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
        ('r', 'u1'),  ('g', 'u1'),  ('b', 'u1'),
    ]))
    xyz = np.stack([data['x'], data['y'], data['z']], axis=1).copy()
    rgb = np.stack([data['r'], data['g'], data['b']], axis=1) / 255.0
    return xyz, rgb


def pca_align(xyz):
    """
    Rotate xyz so that:
      axis 0 (col 0) = smallest-variance direction  → depth (X)
      axis 1 (col 1) = largest-variance direction   → along street (Y)
      axis 2 (col 2) = mid-variance direction        → height (Z)

    Signs are fixed so that:
      X: positive = mean positive (scanner side)
      Y: positive = majority of span to the right
      Z: positive = upward (median of top-half > bottom-half)

    Returns aligned xyz and the rotation matrix R such that aligned = (xyz - mean) @ R.
    """
    centroid = xyz.mean(axis=0)
    centered = xyz - centroid

    # PCA via SVD
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Vt rows are principal components, sorted largest→smallest variance
    # S[0] > S[1] > S[2]  →  PC0=street, PC1=height, PC2=depth

    pc_street = Vt[0]   # largest variance  → Y
    pc_height = Vt[1]   # mid variance      → Z
    pc_depth  = Vt[2]   # smallest variance → X

    # Build rotation matrix: columns are [depth_axis, street_axis, height_axis]
    R = np.column_stack([pc_depth, pc_street, pc_height])   # shape (3,3)

    aligned = centered @ R   # (N,3) — now col0=X, col1=Y, col2=Z

    # Fix sign ambiguity ─────────────────────────────────────────────────────
    # X (depth): most points should be at positive X (wall is in front)
    if np.median(aligned[:, 0]) < 0:
        R[:, 0] *= -1
        aligned[:, 0] *= -1

    # Y (street): arbitrary; keep positive direction = the larger half of range
    # (no strong prior, just make median positive for consistency)
    if np.median(aligned[:, 1]) < 0:
        R[:, 1] *= -1
        aligned[:, 1] *= -1

    # Z (height): the upper half of the facade must have larger Z values.
    # Proxy: mean Z of top-50% in raw Z should be > mean Z of bottom-50%.
    z_median = np.median(aligned[:, 2])
    mean_upper = aligned[aligned[:, 2] > z_median, 2].mean()
    mean_lower = aligned[aligned[:, 2] < z_median, 2].mean()
    if mean_upper < mean_lower:   # Z is flipped — up is negative
        R[:, 2] *= -1
        aligned[:, 2] *= -1

    return aligned, centroid, R


def stat_outlier_removal(xyz, rgb, k=20, thresh=2.0):
    """Remove points whose mean distance to k nearest neighbours > thresh * global_mean."""
    from scipy.spatial import KDTree
    tree = KDTree(xyz)
    dists, _ = tree.query(xyz, k=k + 1)   # first neighbour is self
    mean_dists = dists[:, 1:].mean(axis=1)
    cutoff = mean_dists.mean() + thresh * mean_dists.std()
    mask = mean_dists < cutoff
    return xyz[mask], rgb[mask]


def print_stats(label, xyz):
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    for i, ax in enumerate(['X (depth) ', 'Y (street)', 'Z (height)']):
        lo, hi = xyz[:, i].min(), xyz[:, i].max()
        print(f"  {ax}  {lo:+8.3f} → {hi:+8.3f}  span {hi-lo:.3f} m")
    print(f"  Points: {len(xyz):,}")


# ── load ────────────────────────────────────────────────────────────────────

print("Loading 3_21_2026.ply …")
xyz_a, rgb_a = load_ply("3_21_2026.ply")
print(f"  {len(xyz_a):,} points")

print("Loading 120thamst.ply …")
xyz_b_raw, rgb_b = load_ply("120thamst.ply")
print(f"  {len(xyz_b_raw):,} points")

# ── PCA-align the second scan ────────────────────────────────────────────────

print("\nRunning PCA alignment on 120thamst.ply …")
xyz_b_aligned, centroid_b, R_b = pca_align(xyz_b_raw)

print(f"  Centroid removed: {centroid_b}")
print(f"  Rotation matrix R (columns = new X/Y/Z in old frame):\n{R_b}")

# Re-centre Z so the floor sits near Z=0 (match convention of first scan)
z_floor_b = np.percentile(xyz_b_aligned[:, 2], 2)   # 2nd percentile ≈ floor
xyz_b_aligned[:, 2] -= z_floor_b

z_floor_a = np.percentile(xyz_a[:, 2], 2)
xyz_a_plot = xyz_a.copy()
xyz_a_plot[:, 2] -= z_floor_a

print_stats("3_21_2026.ply  (original, floor-zeroed)", xyz_a_plot)
print_stats("120thamst.ply  (PCA-aligned, floor-zeroed)", xyz_b_aligned)

# ── optional: statistical outlier removal ────────────────────────────────────
# Commented out by default — uncomment if you want cleaner clouds before plotting
# xyz_b_aligned, rgb_b = stat_outlier_removal(xyz_b_aligned, rgb_b)

# ── side-by-side audit plot ──────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Side-by-side audit — same coordinate convention\n"
             "Row 1: 3_21_2026.ply (original)   "
             "Row 2: 120thamst.ply (PCA-aligned)", fontsize=11)

datasets = [
    ("3_21_2026.ply",      xyz_a_plot,      rgb_a),
    ("120thamst.ply (PCA)", xyz_b_aligned,  rgb_b),
]

views = [
    ("Top-down",      1, 0, "Y — along street (m)", "X — depth (m)"),
    ("Front (facade)",1, 2, "Y — along street (m)", "Z — height (m)"),
    ("Side profile",  0, 2, "X — depth (m)",         "Z — height (m)"),
]

for row, (label, xyz, rgb) in enumerate(datasets):
    for col, (title, xi, yi, xlabel, ylabel) in enumerate(views):
        ax = axes[row, col]
        # subsample for speed
        idx = np.random.choice(len(xyz), min(len(xyz), 40_000), replace=False)
        ax.scatter(xyz[idx, xi], xyz[idx, yi], c=rgb[idx], s=0.3, alpha=0.5)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_aspect('equal')
        if row == 0:
            ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)

# row labels
for row, (label, *_) in enumerate(datasets):
    axes[row, 0].set_ylabel(f"{label}\n{axes[row,0].get_ylabel()}", fontsize=8)

plt.tight_layout()
out = "aligned_audit.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"\nSaved {out}")
