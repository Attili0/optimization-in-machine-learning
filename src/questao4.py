"""Questão 4: Cálculo Diferenciável, Gradientes, Hessiana e Regularização Ridge."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from src.datasets import get_system_b


def loss_q4(beta: np.ndarray, B: np.ndarray, y: np.ndarray) -> float:
    """Perda quadrática não regularizada L(beta) = ||B beta - y||_2^2."""
    r = B @ beta - y
    return float(np.dot(r, r))


def loss_ridge_q4(beta: np.ndarray, B: np.ndarray, y: np.ndarray, lam: float = 1.0) -> float:
    """Perda quadrática regularizada Ridge L_lam(beta) = ||B beta - y||_2^2 + lam * ||beta||_2^2."""
    return loss_q4(beta, B, y) + float(lam * np.dot(beta, beta))


def grad_q4(beta: np.ndarray, B: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradiente analítico 2 B^T (B beta - y)."""
    return 2.0 * (B.T @ (B @ beta - y))


def central_finite_diff_grad(
    beta: np.ndarray, B: np.ndarray, y: np.ndarray, h: float = 1e-5
) -> np.ndarray:
    """Gradiente numérico por diferença finita central (h = 1e-5)."""
    d = len(beta)
    g_fd = np.zeros(d, dtype=np.float64)
    for j in range(d):
        e_j = np.zeros(d, dtype=np.float64)
        e_j[j] = 1.0
        g_fd[j] = (loss_q4(beta + h * e_j, B, y) - loss_q4(beta - h * e_j, B, y)) / (2.0 * h)
    return g_fd


def run_questao4(save_figures: bool = True) -> dict:
    """Executa todas as validações da Questão 4 e gera figuras."""
    B, yB, beta0 = get_system_b()

    L0 = loss_q4(beta0, B, yB)
    g0_ana = grad_q4(beta0, B, yB)
    g0_fd = central_finite_diff_grad(beta0, B, yB, h=1e-5)
    max_diff_fd = float(np.max(np.abs(g0_ana - g0_fd)))
    assert max_diff_fd < 1e-9, f"Diferença finita divergente: {max_diff_fd}"

    u_star = -g0_ana / np.linalg.norm(g0_ana)

    # Hessiana não regularizada 2 B^T B e regularizada 2(B^T B + I)
    H_unreg = 2.0 * (B.T @ B)
    H_reg = 2.0 * (B.T @ B + 1.0 * np.eye(3))
    eigs_unreg = np.linalg.eigvalsh(H_unreg)
    eigs_reg = np.linalg.eigvalsh(H_reg)
    assert np.allclose(eigs_unreg, [0.0, 6.0, 42.0], atol=1e-12)
    assert np.allclose(eigs_reg, [2.0, 8.0, 44.0], atol=1e-12)

    # Solução particular das Equações Normais e solução Ridge (lambda = 1)
    beta_star, _, _, _ = np.linalg.lstsq(B, yB, rcond=None)
    beta_ridge = np.linalg.solve(B.T @ B + 1.0 * np.eye(3), B.T @ yB)
    n_null = np.array([1.0, 1.0, -1.0], dtype=np.float64)
    assert np.allclose(B @ n_null, 0.0, atol=1e-14)

    # Varredura ao longo do vale plano t in [-2, 2]
    t_vals = np.linspace(-2.0, 2.0, 17)
    L_valley = np.array([loss_q4(beta_star + t * n_null, B, yB) for t in t_vals])
    L_ridge_valley = np.array([loss_ridge_q4(beta_ridge + t * n_null, B, yB, lam=1.0) for t in t_vals])

    # Comprovação empírica de invariância e curvatura lambda * ||n||_2^2 = 3 (para L_1 = ||B beta - y||^2 + ||beta||^2)
    assert np.allclose(L_valley, L_valley[0], atol=1e-13)
    expected_ridge_curve = loss_ridge_q4(beta_ridge, B, yB, lam=1.0) + 1.0 * (np.dot(n_null, n_null)) * (t_vals ** 2)
    assert np.allclose(L_ridge_valley, expected_ridge_curve, atol=1e-12)

    print("=== QUESTÃO 4: CÁLCULO DIFERENCIÁVEL, HESSIANA E VALES PLANOS (RIDGE) ===")
    print(f"Em beta_0=[1,1,1]^T: L(beta_0) = {L0:.4f} | grad_analitico = {g0_ana} | u* = {np.round(u_star, 6)}")
    print(f"Diferença máxima ||grad_analitico - grad_FD(h=1e-5)||_inf = {max_diff_fd:.3e}")
    print(f"Autovalores Hessiana 2 B^T B: {np.round(eigs_unreg, 6)}")
    print(f"Autovalores Hessiana Ridge 2(B^T B + I): {np.round(eigs_reg, 6)}")
    print(f"Solução OLS norma mínima beta* = {beta_star} | L(beta*) = {L_valley[0]:.6f}")
    print(f"Solução Ridge (lambda=1) beta_ridge = {beta_ridge} | L_1(beta_ridge) = {loss_ridge_q4(beta_ridge, B, yB):.6f}")
    print("Invariância no vale plano confirmada: max|L(beta* + t*n) - L(beta*)| =", float(np.max(np.abs(L_valley - L_valley[0]))))
    print("Tabela ao longo do núcleo n=[1,1,-1]^T (t in [-2, 2]):")
    for t_val, lv, lrv in zip(t_vals[::4], L_valley[::4], L_ridge_valley[::4]):
        print(f"  t = {t_val:+5.2f} | L(beta*+tn) = {lv:.6f} | L_1(beta_1+tn) = {lrv:.6f} (diff = {lrv - L_ridge_valley[len(t_vals)//2]:.6f})")
    print()

    if save_figures:
        os.makedirs("figures", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        t_fine = np.linspace(-2.0, 2.0, 200)
        axes[0].plot(t_fine, [loss_q4(beta_star + t * n_null, B, yB) for t in t_fine], lw=2.5, label="Sem Regularização L(beta* + t n)")
        axes[0].plot(t_fine, [loss_ridge_q4(beta_ridge + t * n_null, B, yB, 1.0) for t in t_fine], lw=2.5, color="#d62728", label="Ridge L_1(beta_1 + t n) [Curvatura = 6]")
        axes[0].set_xlabel("Deslocamento t ao longo do núcleo n = [1, 1, -1]^T")
        axes[0].set_ylabel("Valor da Função de Perda")
        axes[0].set_title("Fechamento Estrito do Vale Plano via Regularização Ridge")
        axes[0].legend(loc="best", fontsize=8.5)

        hs = np.logspace(-15, -1, 60)
        errs_fd = [np.max(np.abs(grad_q4(beta0, B, yB) - central_finite_diff_grad(beta0, B, yB, h=h_val))) for h_val in hs]
        axes[1].loglog(hs, errs_fd, marker="o", markersize=3, lw=1.5, color="#2ca02c")
        axes[1].set_xlabel("Passo h da Diferença Finita Central")
        axes[1].set_ylabel("Erro Absoluto Máximo ||g_ana - g_FD||_inf")
        axes[1].set_title("Cancelamento Catastrófico O(eps_maq / h) em Função Quadrática")
        plt.tight_layout()
        plt.savefig("figures/q4_vale_plano_vs_ridge.png", dpi=300)
        plt.close(fig)

    return {"beta_star": beta_star, "beta_ridge": beta_ridge, "max_diff_fd": max_diff_fd}


if __name__ == "__main__":
    run_questao4()

