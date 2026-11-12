"""Questão 1: Normas, Métricas, Escala e o Algoritmo k-NN em NumPy Puro."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from src.datasets import get_dataset_d


def compute_distances_vectorized(
    X: np.ndarray, x_query: np.ndarray
) -> dict[str, np.ndarray]:
    """Computa de forma totalmente vetorizada as distâncias L1, L2, Linf e Mahalanobis."""
    diff = X - x_query[None, :]
    d_l1 = np.sum(np.abs(diff), axis=1)
    d_l2 = np.sqrt(np.sum(diff ** 2, axis=1))
    d_linf = np.max(np.abs(diff), axis=1)

    S_cov = np.cov(X, rowvar=False, ddof=1)
    S_inv = np.linalg.inv(S_cov)
    d_mahal = np.sqrt(np.einsum("ij,jk,ik->i", diff, S_inv, diff))

    return {
        "L1": d_l1,
        "L2": d_l2,
        "Linf": d_linf,
        "Mahalanobis": d_mahal,
    }


def zscore_standardize(
    X: np.ndarray, ddof: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Padroniza X via Z-score usando desvio-padrão com ddof especificado."""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=ddof)
    return (X - mu) / sigma, mu, sigma


def minmax_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Escalonamento Min-Max para [0, 1]."""
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    return (X - x_min) / (x_max - x_min), x_min, x_max


def cosine_dissimilarity(u: np.ndarray, v: np.ndarray) -> float:
    """Calcula a dissimilaridade do cosseno d_cos(u, v) = 1 - <u, v> / (||u||_2 ||v||_2)."""
    num = float(np.dot(u, v))
    den = float(np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v)))
    return 1.0 - (num / den)


def knn_predict(
    distances: np.ndarray,
    y_reg: np.ndarray,
    c_cls: np.ndarray,
    k: int,
) -> tuple[int, float, np.ndarray]:
    """Retorna a classe (voto majoritário), o valor de regressão (média) e os índices dos k vizinhos."""
    nn_idx = np.argsort(distances)[:k]
    votes = c_cls[nn_idx]
    pred_class = int(np.bincount(votes).argmax())
    pred_reg = float(np.mean(y_reg[nn_idx]))
    return pred_class, pred_reg, nn_idx


def run_questao1(save_figures: bool = True) -> dict:
    """Executa todas as validações da Questão 1 e gera figuras."""
    X_data, y_prices, c_labels, x_novo = get_dataset_d()
    dists = compute_distances_vectorized(X_data, x_novo)

    # Validação do contraexemplo da dissimilaridade do cosseno em R^2
    u = np.array([1.0, 0.0])
    v = np.array([np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0])
    w = np.array([0.0, 1.0])
    d_uv = cosine_dissimilarity(u, v)
    d_vw = cosine_dissimilarity(v, w)
    d_uw = cosine_dissimilarity(u, w)
    assert d_uw > d_uv + d_vw, "Falha no contraexemplo da desigualdade triangular do cosseno!"

    # Predições k-NN nos dados brutos sob L2
    c_k1, _, idx_k1 = knn_predict(dists["L2"], y_prices, c_labels, k=1)
    c_k3, y_k3, idx_k3 = knn_predict(dists["L2"], y_prices, c_labels, k=3)
    assert c_k1 == 0 and c_k3 == 0, "Erro na classificação k-NN bruta!"
    assert np.isclose(y_k3, 8.9 / 3.0, atol=1e-12), "Erro na regressão k-NN (k=3)!"

    # Comparação com Z-score (ddof=1)
    X_z, mu, sigma = zscore_standardize(X_data, ddof=1)
    x_novo_z = (x_novo - mu) / sigma
    dists_z = compute_distances_vectorized(X_z, x_novo_z)
    c_k1_z, _, idx_k1_z = knn_predict(dists_z["L2"], y_prices, c_labels, k=1)
    c_k3_z, y_k3_z, idx_k3_z = knn_predict(dists_z["L2"], y_prices, c_labels, k=3)

    print("=== QUESTÃO 1: NORMAS, MÉTRICAS, ESCALA E k-NN ===")
    print("Amostra |   d_L1 (Rank) |   d_L2 (Rank) | d_Linf (Rank) | d_L2_Z (Rank) | d_Mahal (Rank)")
    r_l1 = np.argsort(np.argsort(dists["L1"])) + 1
    r_l2 = np.argsort(np.argsort(dists["L2"])) + 1
    r_li = np.argsort(np.argsort(dists["Linf"])) + 1
    r_lz = np.argsort(np.argsort(dists_z["L2"])) + 1
    r_ma = np.argsort(np.argsort(dists["Mahalanobis"])) + 1
    for i in range(len(X_data)):
        print(
            f"   {i+1}    | {dists['L1'][i]:6.2f} ({r_l1[i]}º)   |"
            f" {dists['L2'][i]:6.4f} ({r_l2[i]}º)  | {dists['Linf'][i]:6.2f} ({r_li[i]}º)   |"
            f" {dists_z['L2'][i]:6.4f} ({r_lz[i]}º)  | {dists['Mahalanobis'][i]:6.4f} ({r_ma[i]}º)"
        )
    print(f"Contraexemplo Cosseno: d_cos(u,w)={d_uw:.4f} > d_cos(u,v)+d_cos(v,w)={d_uv+d_vw:.4f}")
    print(f"k-NN Bruto (L2): k=1 -> classe {c_k1} (vizinho {idx_k1+1}); k=3 -> classe {c_k3}, y_hat={y_k3:.6f} (vizinhos {idx_k3+1})")
    print(f"k-NN Z-Score (L2): k=1 -> classe {c_k1_z} (vizinho {idx_k1_z+1}); k=3 -> classe {c_k3_z}, y_hat={y_k3_z:.6f} (vizinhos {idx_k3_z+1})\n")

    if save_figures:
        os.makedirs("figures", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        # Painel 1: Espaço Bruto
        ax = axes[0]
        for cls, color, label in [(0, "#1f77b4", "Classe 0"), (1, "#d62728", "Classe 1")]:
            mask = c_labels == cls
            ax.scatter(X_data[mask, 0], X_data[mask, 1], c=color, s=80, label=label, zorder=3)
        for i, (x1, x2) in enumerate(X_data, 1):
            ax.annotate(f"#{i}", (x1 + 0.15, x2 + 0.4), fontsize=9)
        ax.scatter(x_novo[0], x_novo[1], c="#2ca02c", marker="*", s=200, label="x_novo", zorder=4)
        circle = plt.Circle(x_novo, dists["L2"][idx_k3[-1]], color="#2ca02c", fill=False, ls="--", lw=1.5, label="Bola L2 (k=3)")
        ax.add_patch(circle)
        ax.set_title("Espaço Bruto (Distorção pela Escala da Idade)")
        ax.set_xlabel("Área x1 (10 m²)")
        ax.set_ylabel("Idade x2 (anos)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", fontsize=8)

        # Painel 2: Espaço Padronizado Z-Score
        ax2 = axes[1]
        for cls, color, label in [(0, "#1f77b4", "Classe 0"), (1, "#d62728", "Classe 1")]:
            mask = c_labels == cls
            ax2.scatter(X_z[mask, 0], X_z[mask, 1], c=color, s=80, label=label, zorder=3)
        for i, (z1, z2) in enumerate(X_z, 1):
            ax2.annotate(f"#{i}", (z1 + 0.06, z2 + 0.06), fontsize=9)
        ax2.scatter(x_novo_z[0], x_novo_z[1], c="#2ca02c", marker="*", s=200, label="x_novo (Z)", zorder=4)
        r_raw = dists["L2"][idx_k3[-1]]
        theta = np.linspace(0, 2 * np.pi, 200)
        ell_z1 = x_novo_z[0] + (r_raw / sigma[0]) * np.cos(theta)
        ell_z2 = x_novo_z[1] + (r_raw / sigma[1]) * np.sin(theta)
        ax2.plot(ell_z1, ell_z2, color="gray", ls=":", lw=1.4, label="Bola Bruta Mapeada (Elipse)")
        circle_z = plt.Circle(x_novo_z, dists_z["L2"][idx_k3_z[-1]], color="#2ca02c", fill=False, ls="--", lw=1.5, label="Bola L2 Z-Score (k=3)")
        ax2.add_patch(circle_z)
        ax2.set_title("Espaço Padronizado Z-Score (Bola Isotrópica)")
        ax2.set_xlabel("Área Padronizada z1")
        ax2.set_ylabel("Idade Padronizada z2")
        ax2.set_aspect("equal", adjustable="datalim")
        ax2.legend(loc="best", fontsize=8)

        plt.tight_layout()
        plt.savefig("figures/q1_bolas_metricas_escala.png", dpi=300)
        plt.close(fig)

    return {"dists": dists, "c_k1": c_k1, "c_k3": c_k3, "y_k3": y_k3}


if __name__ == "__main__":
    run_questao1()

