"""Questão 6: Otimização Estocástica, Aceleração por Momento, Adam e Método de Newton."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from src.datasets import get_dataset_d


def run_questao6(save_figures: bool = True) -> dict:
    """Executa as três simulações obrigatórias da Parte E da Questão 6."""
    X_data, y_prices, _, _ = get_dataset_d()
    N = len(X_data)
    mu = np.mean(X_data, axis=0)
    sigma = np.std(X_data, axis=0, ddof=1)
    X_tilde = np.column_stack([np.ones(N), (X_data - mu) / sigma])
    w_ols = np.linalg.solve(X_tilde.T @ X_tilde, X_tilde.T @ y_prices)

    # Parte E.1: SGD com Mini-Batching (B in {1, 2, 6}) e decaimento de Robbins-Monro (200 épocas)
    sgd_histories: dict[int, list[float]] = {}
    sgd_final_weights: dict[int, np.ndarray] = {}
    for B_size in [1, 2, 6]:
        rng = np.random.default_rng(42)
        w = np.zeros(3, dtype=np.float64)
        alpha0, gamma = 0.20, 0.01
        t = 0
        epoch_losses = []
        n_steps_per_epoch = N // B_size
        for _ in range(200):
            loss_val = float(0.5 * np.mean((X_tilde @ w - y_prices) ** 2))
            epoch_losses.append(loss_val)
            for _ in range(n_steps_per_epoch):
                idx = rng.choice(N, size=B_size, replace=True)
                xb, yb = X_tilde[idx], y_prices[idx]
                grad = (xb.T @ (xb @ w - yb)) / B_size
                alpha_t = alpha0 / (1.0 + gamma * t)
                w -= alpha_t * grad
                t += 1
        sgd_histories[B_size] = epoch_losses
        sgd_final_weights[B_size] = w.copy()

    # Parte E.2 & E.3: Quadrática mal condicionada f(w1, w2) = 0.5*(100*w1^2 + w2^2) (kappa = 100)
    def grad_quad(w: np.ndarray) -> np.ndarray:
        return np.array([100.0 * w[0], 1.0 * w[1]], dtype=np.float64)

    w0 = np.array([10.0, 10.0], dtype=np.float64)
    tol = 1e-6

    # 1. GD Padrão (Chebyshev alpha* = 2/101)
    alpha_cheb = 2.0 / 101.0
    w = w0.copy()
    traj_gd = [w.copy()]
    while np.linalg.norm(w) >= tol and len(traj_gd) < 10000:
        w = w - alpha_cheb * grad_quad(w)
        traj_gd.append(w.copy())

    # 2. Momento de Polyak (alpha=0.01, beta=0.9)
    w = w0.copy()
    v = np.zeros(2)
    traj_polyak = [w.copy()]
    while np.linalg.norm(w) >= tol and len(traj_polyak) < 10000:
        v = 0.9 * v + 0.01 * grad_quad(w)
        w = w - v
        traj_polyak.append(w.copy())

    # 3. Momento Acelerado de Nesterov - NAG (alpha=0.01, beta=0.9)
    w = w0.copy()
    v = np.zeros(2)
    traj_nag = [w.copy()]
    while np.linalg.norm(w) >= tol and len(traj_nag) < 10000:
        v = 0.9 * v + 0.01 * grad_quad(w - 0.9 * v)
        w = w - v
        traj_nag.append(w.copy())

    # 4. Adam (alpha=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
    w = w0.copy()
    m = np.zeros(2)
    v_adam = np.zeros(2)
    traj_adam = [w.copy()]
    t_step = 0
    while np.linalg.norm(w) >= tol and len(traj_adam) < 10000:
        t_step += 1
        g = grad_quad(w)
        m = 0.9 * m + 0.1 * g
        v_adam = 0.999 * v_adam + 0.001 * (g ** 2)
        m_hat = m / (1.0 - 0.9 ** t_step)
        v_hat = v_adam / (1.0 - 0.999 ** t_step)
        w = w - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
        traj_adam.append(w.copy())

    # 5. Método de Newton Exato (Hessiana H = diag(100, 1))
    H_inv = np.diag([1.0 / 100.0, 1.0])
    w_newton = w0 - H_inv @ grad_quad(w0)
    traj_newton = [w0.copy(), w_newton.copy()]
    assert np.allclose(w_newton, 0.0, atol=1e-15), "Newton não convergiu em 1 passo!"

    iters = {
        "GD_Chebyshev": len(traj_gd) - 1,
        "Polyak": len(traj_polyak) - 1,
        "NAG": len(traj_nag) - 1,
        "Adam": len(traj_adam) - 1,
        "Newton": len(traj_newton) - 1,
    }

    print("=== QUESTÃO 6: OTIMIZAÇÃO ESTOCÁSTICA, MOMENTO, ADAM E NEWTON ===")
    print(f"Solução Analítica OLS (Padronizada): w_ols = {w_ols}")
    for B_size in [1, 2, 6]:
        err_b = float(np.linalg.norm(sgd_final_weights[B_size] - w_ols))
        print(f"  SGD Mini-Batch B={B_size}: w_final = {sgd_final_weights[B_size]} | ||w - w_ols||_2 = {err_b:.6f}")
    print("Iterações até ||w_t||_2 < 1e-6 na quadrática f(w1,w2) = 0.5*(100*w1^2 + w2^2):")
    for k_name, n_it in iters.items():
        print(f"  {k_name:15s}: {n_it} iterações")
    print()

    if save_figures:
        os.makedirs("figures", exist_ok=True)
        # Figura 1: SGD Mini-Batching
        fig1, ax1 = plt.subplots(figsize=(7.5, 4.2))
        for B_size, color in [(1, "#d62728"), (2, "#ff7f0e"), (6, "#1f77b4")]:
            ax1.plot(sgd_histories[B_size], label=f"SGD Mini-Batch B = {B_size}", lw=1.8, color=color, alpha=0.9)
        loss_opt = float(0.5 * np.mean((X_tilde @ w_ols - y_prices) ** 2))
        ax1.axhline(loss_opt, color="black", ls="--", lw=1.5, label=f"Ótimo Analítico OLS ({loss_opt:.4f})")
        ax1.set_yscale("log")
        ax1.set_xlabel("Época (0 a 200)")
        ax1.set_ylabel("Perda Quadrática Média (escala log)")
        ax1.set_title("SGD com Passo de Robbins-Monro variando Tamanho de Lote B")
        ax1.legend(loc="best", fontsize=8.5)
        plt.tight_layout()
        plt.savefig("figures/q6_minibatch_sgd_robbins_monro.png", dpi=300)
        plt.close(fig1)

        # Figura 2: Trajetórias e Convergência na Quadrática Mal Condicionada
        fig2, axes2 = plt.subplots(1, 2, figsize=(11.5, 4.5))
        t_gd = np.array(traj_gd)
        t_pol = np.array(traj_polyak)
        t_nag = np.array(traj_nag)
        t_adm = np.array(traj_adam)
        t_new = np.array(traj_newton)

        axes2[0].plot(t_gd[:40, 0], t_gd[:40, 1], "o-", ms=3, lw=1.2, label=f"GD Chebyshev ({iters['GD_Chebyshev']} it)")
        axes2[0].plot(t_pol[:40, 0], t_pol[:40, 1], "s-", ms=3, lw=1.2, label=f"Polyak ({iters['Polyak']} it)")
        axes2[0].plot(t_nag[:40, 0], t_nag[:40, 1], "^-", ms=3, lw=1.2, label=f"NAG ({iters['NAG']} it)")
        axes2[0].plot(t_adm[:80, 0], t_adm[:80, 1], ".-", ms=3, lw=1.2, label=f"Adam ({iters['Adam']} it)")
        axes2[0].plot(t_new[:, 0], t_new[:, 1], "k*--", ms=8, lw=2.0, label="Newton Exato (1 it)")
        axes2[0].set_xlabel("w_1 (Eixo Íngreme: lambda_1 = 100)")
        axes2[0].set_ylabel("w_2 (Eixo Suave: lambda_2 = 1)")
        axes2[0].set_title("Trajetórias no Plano (w_1, w_2) [Primeiros Passos]")
        axes2[0].legend(loc="best", fontsize=8)

        for name, tr in [("GD Chebyshev", t_gd), ("Polyak", t_pol), ("NAG", t_nag), ("Adam", t_adm), ("Newton", t_new)]:
            norms = np.linalg.norm(tr, axis=1)
            axes2[1].semilogy(np.maximum(norms, 1e-16), lw=2, label=name)
        axes2[1].axhline(1e-6, color="gray", ls=":", label="Tolerância 1e-6")
        axes2[1].set_xlim(0, 400)
        axes2[1].set_xlabel("Iteração t")
        axes2[1].set_ylabel("Norma do Erro ||w_t||_2 (escala log)")
        axes2[1].set_title("Velocidade de Convergência até ||w_t||_2 < 1e-6")
        axes2[1].legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig("figures/q6_quadratica_trajetorias_otimizadores.png", dpi=300)
        plt.close(fig2)

    return {"iters": iters, "sgd_final_weights": sgd_final_weights}


if __name__ == "__main__":
    run_questao6()

