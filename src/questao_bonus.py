"""Questão Bônus: Otimização Não-Suave, Operadores Proximais e ISTA/FISTA."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from src.datasets import get_sparse_synthetic_data


def soft_thresholding(v: np.ndarray, thresh: float) -> np.ndarray:
    """Operador de encolhimento suave S_thresh(v) = sign(v) * max(|v| - thresh, 0)."""
    return np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)


def lasso_objective(w: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float) -> float:
    """Objetivo composto F(w) = 0.5 * ||X w - y||_2^2 + lam * ||w||_1."""
    r = X @ w - y
    return float(0.5 * np.dot(r, r) + lam * np.sum(np.abs(w)))


def run_ista(
    X: np.ndarray, y: np.ndarray, lam: float = 5.0, n_iters: int = 500
) -> tuple[np.ndarray, list[float]]:
    """Executa o algoritmo ISTA com passo 1 / L_f, onde L_f = lambda_max(X^T X)."""
    L_f = float(np.max(np.linalg.eigvalsh(X.T @ X)))
    w = np.zeros(X.shape[1], dtype=np.float64)
    hist = []
    for _ in range(n_iters):
        hist.append(lasso_objective(w, X, y, lam))
        grad = X.T @ (X @ w - y)
        w = soft_thresholding(w - (1.0 / L_f) * grad, lam / L_f)
    return w, hist


def run_fista(
    X: np.ndarray, y: np.ndarray, lam: float = 5.0, n_iters: int = 500
) -> tuple[np.ndarray, list[float]]:
    """Executa o algoritmo FISTA (com aceleração de Nesterov O(1/t^2))."""
    L_f = float(np.max(np.linalg.eigvalsh(X.T @ X)))
    w_prev = np.zeros(X.shape[1], dtype=np.float64)
    z = w_prev.copy()
    t_k = 1.0
    hist = []
    for _ in range(n_iters):
        hist.append(lasso_objective(w_prev, X, y, lam))
        grad_z = X.T @ (X @ z - y)
        w_next = soft_thresholding(z - (1.0 / L_f) * grad_z, lam / L_f)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * (t_k ** 2)))
        z = w_next + ((t_k - 1.0) / t_next) * (w_next - w_prev)
        w_prev, t_k = w_next, t_next
    return w_prev, hist


def run_questao_bonus(save_figures: bool = True) -> dict:
    """Executa a validação numérica da Questão Bônus (N=50, d=10)."""
    X_syn, y_syn, w_true = get_sparse_synthetic_data(n_samples=50, n_features=10, noise_std=0.05, seed=42)
    lam_l1 = 5.0
    w_ista, hist_ista = run_ista(X_syn, y_syn, lam=lam_l1, n_iters=400)
    w_fista, hist_fista = run_fista(X_syn, y_syn, lam=lam_l1, n_iters=400)

    supp_true = set(np.nonzero(w_true)[0].tolist())
    supp_ista = set(np.nonzero(np.abs(w_ista) > 1e-8)[0].tolist())
    assert supp_ista == supp_true, f"Suporte recuperado {supp_ista} difere do verdadeiro {supp_true}!"
    assert np.all(w_ista[3:] == 0.0), "Atributos ruidosos não foram exatamente zerados!"

    print("=== QUESTÃO BÔNUS: OTIMIZAÇÃO NÃO-SUAVE E ALGORITMO ISTA/FISTA ===")
    print(f"w_true   = {w_true}")
    print(f"w_ista   = {np.round(w_ista, 4)}")
    print(f"w_fista  = {np.round(w_fista, 4)}")
    print(f"Suporte verdadeiro: {sorted(supp_true)} | Suporte recuperado pelo ISTA: {sorted(supp_ista)}")
    print(f"Coeficientes nulos exatos (j >= 3): {np.sum(w_ista[3:] == 0.0)} de 7\n")

    if save_figures:
        os.makedirs("figures", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        idx = np.arange(len(w_true))
        axes[0].bar(idx - 0.18, w_true, width=0.35, label="w_true (Verdadeiro)", color="#1f77b4")
        axes[0].bar(idx + 0.18, w_ista, width=0.35, label="w_ISTA (Recuperado)", color="#ff7f0e")
        axes[0].set_xticks(idx)
        axes[0].set_xlabel("Índice do Atributo j (0 a 9)")
        axes[0].set_ylabel("Valor do Coeficiente w_j")
        axes[0].set_title("Recuperação Exata de Suporte Esparso via ISTA")
        axes[0].legend(loc="best", fontsize=8.5)

        f_star = min(min(hist_ista), min(hist_fista))
        axes[1].semilogy(np.maximum(np.array(hist_ista) - f_star, 1e-15), lw=2, label="ISTA O(1/t)")
        axes[1].semilogy(np.maximum(np.array(hist_fista) - f_star, 1e-15), lw=2, label="FISTA Acelerado O(1/t^2)")
        axes[1].set_xlim(0, 120)
        axes[1].set_xlabel("Iteração t")
        axes[1].set_ylabel("Subotimalidade F(w_t) - F* (escala log)")
        axes[1].set_title("Convergência ISTA vs. FISTA")
        axes[1].legend(loc="best", fontsize=8.5)
        plt.tight_layout()
        plt.savefig("figures/qbonus_ista_fista_esparsidade.png", dpi=300)
        plt.close(fig)

    return {"w_true": w_true, "w_ista": w_ista, "w_fista": w_fista}


if __name__ == "__main__":
    run_questao_bonus()

