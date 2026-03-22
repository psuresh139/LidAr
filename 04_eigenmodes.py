"""
Step 3: Laplace-Beltrami eigendecomposition of both facade meshes.

  L φ_n = -λ_n M φ_n   (generalised eigenproblem, mass-weighted)

Uses robust_laplacian (Sharp & Crane 2020) which builds a cotangent-weight
Laplacian + lumped mass matrix robust to non-manifold / near-degenerate geometry.

Solves for the K smallest non-trivial eigenpairs with scipy eigsh (ARPACK).
Visualises each eigenmode as a heatmap over the Y-Z facade plane.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import robust_laplacian
import scipy.sparse.linalg as spla
import scipy.sparse as sp
import time

K = 24   # number of non-trivial modes to compute (skip the constant mode λ₀≈0)

# ── load meshes ───────────────────────────────────────────────────────────────

def load_mesh(path):
    d = np.load(path)
    return d['verts'], d['faces'].astype(int), d['vcolors']

print("Loading meshes…")
verts_a, faces_a, vcol_a = load_mesh("mesh_a.npz")
verts_b, faces_b, vcol_b = load_mesh("mesh_b.npz")

# ── Laplace-Beltrami + eigsh ──────────────────────────────────────────────────

def compute_eigenmodes(verts, faces, k, label):
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  {len(verts):,} vertices   {len(faces):,} faces")
    print(f"{'─'*55}")

    t0 = time.time()
    print("  Building Laplace-Beltrami operator…")
    L, M = robust_laplacian.mesh_laplacian(verts, faces)
    print(f"  L: {L.shape}  nnz={L.nnz:,}   M: {M.shape}  ({time.time()-t0:.1f}s)")

    # Shift to avoid the near-zero constant mode interfering with ARPACK
    # Solve (L + σM)φ = λ M φ with σ small negative shift so λ₀ moves away from 0
    sigma = -1e-6

    print(f"  Solving eigsh for {k+1} modes (including constant)…")
    t1 = time.time()
    try:
        # which='SM' = smallest magnitude; sigma-shift makes ARPACK find near-zero eigs
        eigenvalues, eigenvectors = spla.eigsh(
            L, k=k+1, M=M, sigma=sigma, which='LM',
            tol=1e-8, maxiter=5000
        )
    except Exception as e:
        print(f"  eigsh with sigma failed ({e}), retrying with which='SM'…")
        eigenvalues, eigenvectors = spla.eigsh(
            L, k=k+1, M=M, which='SM', tol=1e-6, maxiter=10000
        )
    print(f"  Done in {time.time()-t1:.1f}s")

    # Sort by eigenvalue (eigsh may not return sorted)
    order = np.argsort(eigenvalues)
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Drop the constant mode (λ ≈ 0)
    # Keep k non-trivial modes starting from index 1
    lam = eigenvalues[1:k+1]
    phi = eigenvectors[:, 1:k+1]   # shape (V, k)

    print(f"  λ₁ = {lam[0]:.4f}   λ_{k} = {lam[-1]:.4f}")
    print(f"  Eigenvalue range: [{lam.min():.4f}, {lam.max():.4f}]")

    return lam, phi


lam_a, phi_a = compute_eigenmodes(verts_a, faces_a, K, "Scan A — 3_21_2026.ply")
lam_b, phi_b = compute_eigenmodes(verts_b, faces_b, K, "Scan B — 120thamst.ply (aligned)")

# Save for downstream use
np.savez("eigenmodes_a.npz", eigenvalues=lam_a, eigenvectors=phi_a,
         verts=verts_a, faces=faces_a, vcolors=vcol_a)
np.savez("eigenmodes_b.npz", eigenvalues=lam_b, eigenvectors=phi_b,
         verts=verts_b, faces=faces_b, vcolors=vcol_b)
print("\nSaved eigenmodes_a.npz and eigenmodes_b.npz")

# ── visualise eigenmodes as Y-Z heatmaps ─────────────────────────────────────

def plot_eigenmodes(lam, phi, verts, vcol, label, outfile, cols=6):
    """
    Each mode φ_n lives on the mesh vertices.
    We display it as a scatter in Y-Z coloured by φ_n value,
    arranged in a grid of subplots.
    """
    nrows = K // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 3.2, nrows * 2.4))
    fig.suptitle(f"Laplace-Beltrami eigenmodes — {label}", fontsize=11, y=1.01)

    Y = verts[:, 1]
    Z = verts[:, 2]

    # Subsample for speed if very large
    idx = np.arange(len(verts))
    if len(verts) > 40_000:
        idx = np.random.choice(len(verts), 40_000, replace=False)

    for n in range(K):
        ax = axes[n // cols, n % cols]
        mode = phi[:, n]
        # Normalise for display: zero-mean, unit max-abs
        mode_disp = mode - mode.mean()
        scale = np.abs(mode_disp).max()
        if scale > 0:
            mode_disp /= scale

        sc = ax.scatter(Y[idx], Z[idx], c=mode_disp[idx],
                        cmap='RdBu_r', s=1.5, vmin=-1, vmax=1, linewidths=0)
        ax.set_title(f"φ_{n+1}   λ={lam[n]:.3f}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outfile}")


plot_eigenmodes(lam_a, phi_a, verts_a, vcol_a,
                "3_21_2026.ply", "eigenmodes_a.png")
plot_eigenmodes(lam_b, phi_b, verts_b, vcol_b,
                "120thamst.ply (aligned)", "eigenmodes_b.png")

# ── eigenvalue spectrum plot ──────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Laplace-Beltrami eigenvalue spectra", fontsize=11)

for ax, lam, label, color in [
    (ax1, lam_a, "Scan A — 3_21_2026", 'steelblue'),
    (ax2, lam_b, "Scan B — 120thamst", 'coral'),
]:
    ax.stem(range(1, K+1), lam, linefmt=color, markerfmt=f'o', basefmt='k-')
    ax.set_xlabel("mode index n"); ax.set_ylabel("λ_n")
    ax.set_title(label)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("eigenvalue_spectra.png", dpi=150)
plt.close()
print("Saved eigenvalue_spectra.png")

print("\nAll done. Next: project a photograph onto eigenfunction basis and compare to DCT.")
