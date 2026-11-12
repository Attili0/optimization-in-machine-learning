"""Questão 5: Convergência da Descida do Gradiente, Condicionamento e Regressão Logística."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from src.datasets import get_dataset_d


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Função sigmoide numericamente estável."""
    z_clip = np.clip(z, -500.0, 500.0)
    return np.where(z_clip >= 0, 1.0 / (1.0 + np.exp(-z_clip)), np.exp(z_clip) / (1.0 + np.exp(z_clip)))


def logistic_loss_reg(w: np.ndarray, X: np.ndarray, c: np.ndarray, lam: float = 0.1) -> float:
    """Perda de Entropia Cruzada Binária com regularização L2 (lam * ||w||_2^2)."""
    p = np.clip(sigmoid(X @ w), 1e-15, 1.0 - 1e-15)
    ce = -np.mean(c * np.log(p) + (1.0 - c) * np.log(1.0 - p))
    return float(ce + lam * np.dot(w, w))


def logistic_grad_reg(w: np.ndarray, X: np.ndarray, c: np.ndarray, lam: float = 0.1) -> np.ndarray:
    """Gradiente (1/N) X^T (p - c) + 2 * lam * w."""
    p = sigmoid(X @ w)
    return (X.T @ (p - c)) / len(c) + 2.0 * lam * w


def logistic_hess_reg(w: np.ndarray, X: np.ndarray, lam: float = 0.1) -> np.ndarray:
    """Hessiana (1/N) X^T S X + 2 * lam * I."""
    p = sigmoid(X @ w)
    s = p * (1.0 - p)
    return ((X.T * s) @ X) / len(X) + 2.0 * lam * np.eye(X.shape[1])


def run_questao5(save_figures: bool = True) -> dict:
    """Executa todas as validações da Questão 5 e gera figuras."""
    X_data, y_prices, c_labels, _ = get_dataset_d()
    N = len(X_data)
    X_raw = np.column_stack([np.ones(N), X_data])

    # Condicionamento da matriz bruta vs padronizada (ddof=1 e ddof=0)
    eigs_raw = np.linalg.eigvalsh(2.0 * (X_raw.T @ X_raw))
    kappa_raw = float(eigs_raw[-1] / eigs_raw[0])

    mu = np.mean(X_data, axis=0)
    s_ddof1 = np.std(X_data, axis=0, ddof=1)
    s_ddof0 = np.std(X_data, axis=0, ddof=0)
    X_tilde_1 = np.column_stack([np.ones(N), (X_data - mu) / s_ddof1])
    X_tilde_0 = np.column_stack([np.ones(N), (X_data - mu) / s_ddof0])

    eigs_t1 = np.linalg.eigvalsh(2.0 * (X_tilde_1.T @ X_tilde_1))
    eigs_t0 = np.linalg.eigvalsh(2.0 * (X_tilde_0.T @ X_tilde_0))
    kappa_t1 = float(eigs_t1[-1] / eigs_t1[0])
    kappa_t0 = float(eigs_t0[-1] / eigs_t0[0])

    # Simulação de 2000 iterações de GD nos dados brutos vs padronizados (OLS e Logística)
    alpha_ols_raw = 2.0 / (eigs_raw[0] + eigs_raw[-1])
    alpha_ols_t1 = 2.0 / (eigs_t1[0] + eigs_t1[-1])
    w_ols_raw = np.zeros(3)
    w_ols_t1 = np.zeros(3)
    hist_ols_raw, hist_ols_t1 = [], []
    for _ in range(2000):
        hist_ols_raw.append(float(np.sum((X_raw @ w_ols_raw - y_prices) ** 2)))
        hist_ols_t1.append(float(np.sum((X_tilde_1 @ w_ols_t1 - y_prices) ** 2)))
        w_ols_raw -= alpha_ols_raw * (2.0 * (X_raw.T @ (X_raw @ w_ols_raw - y_prices)))
        w_ols_t1 -= alpha_ols_t1 * (2.0 * (X_tilde_1.T @ (X_tilde_1 @ w_ols_t1 - y_prices)))

    # Treinamento da Regressão Logística Regularizada (lambda = 0.1) por 2000 iterações
    lam = 0.1
    L_log_raw = 0.25 * float(np.max(np.linalg.eigvalsh((X_raw.T @ X_raw) / N))) + 2.0 * lam
    L_log_t1 = 0.25 * float(np.max(np.linalg.eigvalsh((X_tilde_1.T @ X_tilde_1) / N))) + 2.0 * lam
    w_log_raw = np.zeros(3)
    w_log_t1 = np.zeros(3)
    hist_log_raw, hist_log_t1 = [], []
    for _ in range(2000):
        hist_log_raw.append(logistic_loss_reg(w_log_raw, X_raw, c_labels, lam))
        hist_log_t1.append(logistic_loss_reg(w_log_t1, X_tilde_1, c_labels, lam))
        w_log_raw -= (1.0 / L_log_raw) * logistic_grad_reg(w_log_raw, X_raw, c_labels, lam)
        w_log_t1 -= (1.0 / L_log_t1) * logistic_grad_reg(w_log_t1, X_tilde_1, c_labels, lam)

    probs_t1 = sigmoid(X_tilde_1 @ w_log_t1)
    preds_t1 = (probs_t1 >= 0.5).astype(int)
    acc_t1 = float(np.mean(preds_t1 == c_labels))
    assert acc_t1 == 1.0, f"Acurácia inesperada na regressão logística: {acc_t1}"

    # Verificação de unicidade e convergência quadrática via Método de Newton
    w_newton = np.zeros(3)
    for _ in range(15):
        g_n = logistic_grad_reg(w_newton, X_tilde_1, c_labels, lam)
        if np.linalg.norm(g_n) < 1e-12:
            break
        H_n = logistic_hess_reg(w_newton, X_tilde_1, lam)
        w_newton -= np.linalg.solve(H_n, g_n)
    assert np.allclose(w_log_t1, w_newton, atol=1e-10), "GD e Newton divergiram no ótimo único!"

    print("=== QUESTÃO 5: CONDICIONAMENTO E REGRESSÃO LOGÍSTICA REGULARIZADA ===")
    print(f"Número de condição bruto kappa(2 X^T X) = {kappa_raw:.4f}")
    print(f"Número de condição Z-score kappa(2 X_tilde^T X_tilde): ddof=1 -> {kappa_t1:.4f} | ddof=0 -> {kappa_t0:.4f}")
    print(f"OLS GD (2000 iterações): Perda Bruta = {hist_ols_raw[-1]:.6f} vs Perda Padronizada = {hist_ols_t1[-1]:.6f} (Ótimo = 0.239147)")
    print(f"Reg. Logística GD (2000 it, lambda=0.1): Perda Bruta = {hist_log_raw[-1]:.6f} vs Padronizada = {hist_log_t1[-1]:.6f}")
    print(f"Regressão Logística (lambda=0.1, padronizado ddof=1): w_hat = {w_log_t1} (||w_GD - w_Newton||_2 = {np.linalg.norm(w_log_t1 - w_newton):.2e})")
    print(f"Probabilidades estimadas: {np.round(probs_t1, 4)} | Classes previstas: {preds_t1} | Acurácia = {acc_t1*100:.1f}%\n")

    if save_figures:
        os.makedirs("figures", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        axes[0].semilogy(np.array(hist_ols_raw) - 0.2391472524554213, label=f"Dados Brutos (kappa={kappa_raw:.0f})", lw=2)
        axes[0].semilogy(np.maximum(np.array(hist_ols_t1) - 0.2391472524554213, 1e-16), label=f"Dados Padronizados Z-Score (kappa={kappa_t1:.2f})", lw=2)
        axes[0].set_xlim(0, 200)
        axes[0].set_xlabel("Iteração k")
        axes[0].set_ylabel("Subotimalidade f(w_k) - f* (escala log)")
        axes[0].set_title("Impacto do Pré-Condicionamento Z-Score (OLS)")
        axes[0].legend(loc="best", fontsize=8.5)

        axes[1].plot(hist_log_raw, label="GD Reg. Logística (Dados Brutos)", lw=2)
        axes[1].plot(hist_log_t1, label="GD Reg. Logística (Dados Padronizados)", lw=2)
        axes[1].set_xlabel("Iteração k (até 2000)")
        axes[1].set_ylabel("Perda de Entropia Cruzada Regularizada L(w)")
        axes[1].set_title("Treinamento da Regressão Logística (lambda = 0.1)")
        axes[1].legend(loc="best", fontsize=8.5)
        plt.tight_layout()
        plt.savefig("figures/q5_condicionamento_gd_logistica.png", dpi=300)
        plt.close(fig)

    return {"kappa_raw": kappa_raw, "kappa_t1": kappa_t1, "w_log_t1": w_log_t1, "probs_t1": probs_t1, "acc_t1": acc_t1}


if __name__ == "__main__":
    run_questao5()

