import numpy as np
import matplotlib.pyplot as plt
import struct
import sys
import os

# --- load ---
path = sys.argv[1] if len(sys.argv) > 1 else "120thamst.ply"

with open(path, 'rb') as f:
    while f.readline().strip() != b'end_header':
        pass
    raw = f.read()

n = len(raw) // 27
data = np.frombuffer(raw[:n*27], dtype=np.dtype([
    ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
    ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
]))

xyz = np.stack([data['x'], data['y'], data['z']], axis=1)
rgb = np.stack([data['r'], data['g'], data['b']], axis=1) / 255.0

print(f"Points loaded: {n:,}")
print(f"X (depth)  {xyz[:,0].min():.2f} to {xyz[:,0].max():.2f} m  —  span {xyz[:,0].max()-xyz[:,0].min():.2f} m")
print(f"Y (along)  {xyz[:,1].min():.2f} to {xyz[:,1].max():.2f} m  —  span {xyz[:,1].max()-xyz[:,1].min():.2f} m")
print(f"Z (height) {xyz[:,2].min():.2f} to {xyz[:,2].max():.2f} m  —  span {xyz[:,2].max()-xyz[:,2].min():.2f} m")

# --- facade plane isolation ---
# wall is where X density peaks — take the middle 80% of X range
x_lo, x_hi = np.percentile(xyz[:,0], 10), np.percentile(xyz[:,0], 90)
facade = xyz[(xyz[:,0] >= x_lo) & (xyz[:,0] <= x_hi)]
facade_rgb = rgb[(xyz[:,0] >= x_lo) & (xyz[:,0] <= x_hi)]
print(f"\nFacade points (X {x_lo:.2f} to {x_hi:.2f} m): {len(facade):,}")

# --- plots ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(os.path.basename(path), fontsize=12)

# 1. top-down view (X vs Y) — shows block length and depth variation
axes[0].scatter(facade[:,1], facade[:,0], c=facade_rgb, s=0.1, alpha=0.5)
axes[0].set_xlabel("Y — along street (m)")
axes[0].set_ylabel("X — depth from scanner (m)")
axes[0].set_title("Top-down view")
axes[0].set_aspect('equal')

# 2. front view (Y vs Z) — the facade as you'd see it straight on
axes[1].scatter(facade[:,1], facade[:,2], c=facade_rgb, s=0.1, alpha=0.5)
axes[1].set_xlabel("Y — along street (m)")
axes[1].set_ylabel("Z — height (m)")
axes[1].set_title("Front view (facade)")
axes[1].set_aspect('equal')

# 3. side profile (X vs Z) — depth variation and height capture
axes[2].scatter(xyz[:,0], xyz[:,2], c=rgb, s=0.1, alpha=0.3)
axes[2].set_xlabel("X — depth (m)")
axes[2].set_ylabel("Z — height (m)")
axes[2].set_title("Side profile")
axes[2].set_aspect('equal')

plt.tight_layout()
plt.savefig("audit_output.png", dpi=150)
plt.show()
print("\nSaved audit_output.png")