"""Questão 3: Projeção Ortogonal, Teorema Espectral e Decomposição SVD em NumPy Puro."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from src.datasets import get_system_a


def run_questao3(save_figures: bool = True) -> dict:
    """Executa todas as validações da Questão 3 e gera figuras."""
    A, yA = get_system_a()
    S = A.T @ A
    Aty = A.T @ yA

    # Parte A: Solução via solve e lstsq
    w_solve = np.linalg.solve(S, Aty)
    w_lstsq, _, _, _ = np.linalg.lstsq(A, yA, rcond=None)
    assert np.allclose(w_solve, w_lstsq, atol=1e-14)

    y_hat = A @ w_solve
    e_hat = yA - y_hat
    ortho_err = float(np.max(np.abs(A.T @ e_hat)))
    assert ortho_err < 1e-14, f"Erro de ortogonalidade na Q3: {ortho_err}"

    P_proj = A @ np.linalg.inv(S) @ A.T
    assert np.allclose(P_proj, P_proj.T, atol=1e-14)
    assert np.allclose(P_proj @ P_proj, P_proj, atol=1e-14)

    # Parte B: Teorema Espectral em S = A^T A
    eigvals, Q = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]
    eigvals, Q = eigvals[idx], Q[:, idx]
    assert np.allclose(eigvals, [3.0, 1.0], atol=1e-14)
    assert np.allclose(Q.T @ Q, np.eye(2), atol=1e-14)
    assert np.allclose(Q @ np.diag(eigvals) @ Q.T, S, atol=1e-14)

    sigmas = np.sqrt(eigvals)
    kappa_S = float(eigvals[0] / eigvals[1])
    kappa_A = float(sigmas[0] / sigmas[1])
    assert np.isclose(kappa_S, kappa_A ** 2, atol=1e-14)

    # Parte C: SVD reduzida manual vs np.linalg.svd
    V = Q
    U = np.column_stack([(1.0 / sigmas[i]) * (A @ V[:, i]) for i in range(2)])
    Sigma = np.diag(sigmas)
    A_rec = U @ Sigma @ V.T
    assert np.allclose(A_rec, A, atol=1e-14)

    U_np, s_np, Vt_np = np.linalg.svd(A, full_matrices=False)
    assert np.allclose(sigmas, s_np, atol=1e-14)
    for i in range(2):
        assert np.allclose(np.abs(U[:, i]), np.abs(U_np[:, i]), atol=1e-14)
        assert np.allclose(np.abs(V[:, i]), np.abs(Vt_np[i, :]), atol=1e-14)

    # Parte D: Aproximação de posto 1 (Eckart-Young-Mirsky) e Decomposição Polar
    A_rank1 = sigmas[0] * np.outer(U[:, 0], V[:, 0])
    err_spectral = float(np.linalg.norm(A - A_rank1, ord=2))
    assert np.isclose(err_spectral, sigmas[1], atol=1e-14)

    O_polar = U @ V.T
    P_polar = V @ Sigma @ V.T
    assert np.allclose(O_polar.T @ O_polar, np.eye(2), atol=1e-14)
    assert np.allclose(P_polar, P_polar.T, atol=1e-14)
    assert np.allclose(O_polar @ P_polar, A, atol=1e-14)

    print("=== QUESTÃO 3: PROJEÇÃO ORTOGONAL, TEOREMA ESPECTRAL E SVD ===")
    print(f"w_hat = {w_solve} | y_hat = {y_hat} | e_hat = {e_hat}")
    print(f"||A^T e_hat||_inf = {ortho_err:.3e} (< 1e-14)")
    print(f"Autovalores de S = A^T A: {eigvals} | Valores Singulares de A: {sigmas}")
    print(f"kappa(S) = {kappa_S:.4f} == [kappa(A)]^2 = {kappa_A**2:.4f}")
    print(f"Aproximação de Posto 1 A^(1):\n{A_rank1}")
    print(f"Erro espectral ||A - A^(1)||_2 = {err_spectral:.10f} == sigma_2 = {sigmas[1]:.10f}")
    print(f"Decomposição Polar O = U V^T:\n{O_polar}\nP = V Sigma V^T:\n{P_polar}\n")

    if save_figures:
        os.makedirs("figures", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        theta = np.linspace(0, 2 * np.pi, 200)
        circle_pts = np.vstack([np.cos(theta), np.sin(theta)])
        stretched_pts = P_polar @ circle_pts

        axes[0].plot(circle_pts[0], circle_pts[1], label="Círculo Unitário em R^2", lw=2)
        axes[0].plot(stretched_pts[0], stretched_pts[1], label="Elipse Deformada por P = V Sigma V^T", lw=2)
        for i in range(2):
            v_vec = V[:, i] * sigmas[i]
            axes[0].arrow(0, 0, v_vec[0], v_vec[1], head_width=0.08, color="#d62728", zorder=5)
        axes[0].set_aspect("equal")
        axes[0].set_title("Ação Geométrica de P (Deformação Principal)")
        axes[0].legend(loc="best", fontsize=8)

        axes[1].bar(["sigma_1 = sqrt(3)", "sigma_2 = 1.0"], sigmas, color=["#1f77b4", "#ff7f0e"], width=0.45)
        axes[1].set_ylabel("Magnitude do Valor Singular")
        axes[1].set_title("Espectro Singular de A e Cota de Eckart-Young")
        plt.tight_layout()
        plt.savefig("figures/q3_svd_projecao_polar.png", dpi=300)
        plt.close(fig)

    return {"w_hat": w_solve, "sigmas": sigmas, "A_rank1": A_rank1, "err_spectral": err_spectral}


if __name__ == "__main__":
    run_questao3()

