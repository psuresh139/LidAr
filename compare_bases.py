"""
Step 4: Compare Laplace-Beltrami eigenfunction basis vs 2D DCT basis.

Test signal: grayscale luminance from the LiDAR point-cloud RGB,
             projected onto the (205×77) facade grid — natively registered,
             no photo alignment needed.

For each K (number of basis functions retained):
  LB:  c_n = φ_n^T M f  (mass-weighted projection)
       f_K = Σ_n c_n φ_n
  DCT: apply 2D type-II DCT, zero all but K lowest-frequency coefficients,
       inverse DCT  (ordering by 2D frequency radius, same as JPEG zigzag)

Metric: normalised reconstruction error  ε(K) = ||f - f_K||² / ||f||²

Also produces:
  - side-by-side facade image vs reconstructions at K=6,12,24
  - error curves for both bases
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.sparse as sp
from scipy.fft import dctn, idctn

# ── load mesh + eigenmodes for scan B ────────────────────────────────────────

print("Loading mesh B and eigenmodes…")
mesh = np.load("mesh_b.npz")
eig  = np.load("eigenmodes_b.npz")

verts   = mesh['verts']        # (V, 3)
vcolors = mesh['vcolors']      # (V, 3)  — per-vertex RGB already gridded

phi = eig['eigenvectors']      # (V, K)
lam = eig['eigenvalues']       # (K,)
K_total = phi.shape[1]         # = 24

# Grid dimensions (from mesh_b: 205 × 77)
# Infer from verts: count unique Y and Z grid values
Y_vals = np.unique(np.round(verts[:, 1], 6))
Z_vals = np.unique(np.round(verts[:, 2], 6))
NY, NZ = len(Y_vals), len(Z_vals)
V = NY * NZ
print(f"Grid: {NY} × {NZ} = {V} vertices")
assert V == len(verts), f"Grid size mismatch: {V} vs {len(verts)}"

# ── build signal f: grayscale luminance on vertices ──────────────────────────

# Luminance (BT.601): L = 0.299R + 0.587G + 0.114B
f_vert = (0.299 * vcolors[:, 0] +
          0.587 * vcolors[:, 1] +
          0.114 * vcolors[:, 2])   # (V,)

# Zero-mean (required for projection; DC component handled separately)
f_mean = f_vert.mean()
f = f_vert - f_mean                # (V,) zero-mean signal

# Reshape to 2D grid for visualisation and DCT
f_2d = f.reshape(NY, NZ)          # (NY, NZ)

# ── rebuild mass matrix (diagonal, lumped) ───────────────────────────────────
# For the cotangent Laplacian on a uniform grid, M ≈ cell_area * I (lumped).
# We rebuild it from robust_laplacian so the inner products are exact.

print("Rebuilding mass matrix…")
import robust_laplacian
faces = mesh['faces'].astype(int)
L_mat, M_mat = robust_laplacian.mesh_laplacian(verts, faces)
# M_mat is diagonal; extract diagonal for fast computation
M_diag = np.array(M_mat.diagonal())   # (V,)

# ── LB projection ────────────────────────────────────────────────────────────

print("Computing LB projections…")
# φ_n are M-orthonormal: φ_n^T M φ_n = 1
# coefficient c_n = φ_n^T M f = (M_diag * f) · φ_n
Mf = M_diag * f                        # (V,)
coeffs_lb = phi.T @ Mf                 # (K_total,)  — one per mode

# Reconstruction error as function of K retained
K_range = np.arange(1, K_total + 1)
norm_f_sq = np.dot(f, Mf)             # ||f||²_M

err_lb = np.zeros(K_total)
f_recon_lb = np.zeros(V)
for k in range(K_total):
    f_recon_lb += coeffs_lb[k] * phi[:, k]
    residual = f - f_recon_lb
    err_lb[k] = np.dot(residual, M_diag * residual) / norm_f_sq

print(f"LB errors: K=1→{err_lb[0]:.4f}  K=6→{err_lb[5]:.4f}  K=24→{err_lb[-1]:.4f}")

# ── DCT projection ────────────────────────────────────────────────────────────

print("Computing DCT projections…")
# 2D type-II DCT (orthonormal normalisation)
F_dct = dctn(f_2d, norm='ortho')      # (NY, NZ) — all coefficients

# Sort DCT coefficients by 2D spatial frequency radius (≈ zigzag ordering)
uu = np.arange(NY)
vv = np.arange(NZ)
UU, VV = np.meshgrid(uu, vv, indexing='ij')
freq_radius = np.sqrt((UU / NY) ** 2 + (VV / NZ) ** 2)   # normalised frequency
flat_order = np.argsort(freq_radius.ravel())    # low-freq first

F_flat = F_dct.ravel()
norm_f_dct_sq = np.sum(f_2d ** 2)     # standard L² norm for DCT

err_dct = np.zeros(K_total)
F_recon_flat = np.zeros(V)
for k in range(K_total):
    idx = flat_order[k]
    F_recon_flat[idx] = F_flat[idx]
    f_recon_dct = idctn(F_recon_flat.reshape(NY, NZ), norm='ortho')
    residual_dct = f_2d - f_recon_dct
    err_dct[k] = np.sum(residual_dct ** 2) / norm_f_dct_sq

print(f"DCT errors: K=1→{err_dct[0]:.4f}  K=6→{err_dct[5]:.4f}  K=24→{err_dct[-1]:.4f}")

# ── visualise reconstructions at K = 6, 12, 24 ───────────────────────────────

snap_K = [6, 12, 24]

fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(3, len(snap_K) + 1, figure=fig, hspace=0.35, wspace=0.15)

# Column 0: original
ax_orig = fig.add_subplot(gs[:, 0])
im = ax_orig.imshow(f_2d.T + f_mean, origin='lower', cmap='gray',
                    aspect='auto', vmin=0, vmax=1)
ax_orig.set_title("Original\n(LiDAR RGB → luma\nnative registration)", fontsize=8)
ax_orig.set_xlabel("Y street →"); ax_orig.set_ylabel("Z height →")
plt.colorbar(im, ax=ax_orig, fraction=0.046, label='luma')

# Recompute full reconstructions for snaps
for col, k in enumerate(snap_K):
    # LB
    f_lb_k = (coeffs_lb[:k] @ phi[:, :k].T)      # (V,)
    # DCT
    F_dct_k = np.zeros(V)
    F_dct_k[flat_order[:k]] = F_flat[flat_order[:k]]
    f_dct_k = idctn(F_dct_k.reshape(NY, NZ), norm='ortho').ravel()

    for row, (signal, label) in enumerate([
        (f_lb_k,  f"LB  K={k}"),
        (f_dct_k, f"DCT K={k}"),
        (f_lb_k - f_dct_k, "LB − DCT\n(difference)"),
    ]):
        ax = fig.add_subplot(gs[row, col + 1])
        if row < 2:
            ax.imshow((signal.reshape(NY, NZ) + f_mean).T,
                      origin='lower', cmap='gray', aspect='auto', vmin=0, vmax=1)
        else:
            vmax = np.abs(signal).max() + 1e-6
            ax.imshow(signal.reshape(NY, NZ).T,
                      origin='lower', cmap='RdBu_r', aspect='auto',
                      vmin=-vmax, vmax=vmax)
        err_l = err_lb[k-1]
        err_d = err_dct[k-1]
        if row == 0:
            ax.set_title(f"{label}\nε={err_l:.4f}", fontsize=8)
        elif row == 1:
            ax.set_title(f"{label}\nε={err_d:.4f}", fontsize=8)
        else:
            ax.set_title(label, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Facade image reconstruction: Laplace-Beltrami vs DCT basis\n"
             "Signal: LiDAR point-cloud luminance on 205×77 facade grid (30 cm/cell)",
             fontsize=10)
plt.savefig("basis_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved basis_comparison.png")

# ── error curve plot ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(K_range, err_lb,  'o-', color='steelblue', label='Laplace-Beltrami (geometry-adapted)')
ax.plot(K_range, err_dct, 's-', color='coral',     label='2D DCT (universal cosines)')
for k in snap_K:
    ax.axvline(k, color='gray', ls=':', lw=0.8)
ax.set_xlabel("K — number of basis functions retained")
ax.set_ylabel("Normalised reconstruction error  ε(K) = ‖f − f_K‖² / ‖f‖²")
ax.set_title("Scan B (120th St facade) — LiDAR luminance reconstruction\n"
             "LB basis vs DCT basis as function of K")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, K_total + 1)
ax.set_ylim(0, 1.05)
# Annotate LB advantage at each snap
for k in snap_K:
    delta = err_dct[k-1] - err_lb[k-1]
    sign = '+' if delta >= 0 else ''
    ax.annotate(f"Δε={sign}{delta:.4f}", xy=(k, err_lb[k-1]),
                xytext=(k + 0.3, err_lb[k-1] + 0.04),
                fontsize=7, color='steelblue')

plt.tight_layout()
plt.savefig("error_curves.png", dpi=150)
plt.close()
print("Saved error_curves.png")

# ── print summary table ───────────────────────────────────────────────────────
print("\n" + "─"*55)
print(f"{'K':>4}  {'ε_LB':>10}  {'ε_DCT':>10}  {'Δ (DCT−LB)':>12}  winner")
print("─"*55)
for k in K_range:
    delta = err_dct[k-1] - err_lb[k-1]
    winner = "LB " if delta > 0 else ("DCT" if delta < 0 else "tie")
    print(f"{k:>4}  {err_lb[k-1]:>10.6f}  {err_dct[k-1]:>10.6f}  {delta:>+12.6f}  {winner}")
