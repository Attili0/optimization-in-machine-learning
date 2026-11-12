"""Módulo de definição canônica e imutável dos conjuntos de dados do projeto."""
from __future__ import annotations

import numpy as np


def get_dataset_d() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Retorna o Dataset Principal D (Avaliação Imobiliária) e o ponto x_novo."""
    X_data = np.array([
        [5.0, 30.0],
        [7.0, 38.0],
        [9.0, 40.0],
        [3.5, 19.0],
        [8.0, 15.0],
        [2.0, 35.0],
    ], dtype=np.float64)
    y_prices = np.array([2.5, 3.8, 4.2, 1.9, 4.5, 1.2], dtype=np.float64)
    c_labels = np.array([0, 0, 1, 0, 1, 0], dtype=np.int64)
    x_novo = np.array([6.0, 20.0], dtype=np.float64)
    return X_data, y_prices, c_labels, x_novo


def get_system_a() -> tuple[np.ndarray, np.ndarray]:
    """Retorna a Matriz-Toy A (3x2) e o vetor de alvos y_A (3,) da Questão 3."""
    A_mat = np.array([
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ], dtype=np.float64)
    y_A = np.array([4.0, 1.0, 1.0], dtype=np.float64)
    return A_mat, y_A


def get_system_b() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna a Matriz-Toy B (3x3), o vetor y_B (3,) e beta_0 (3,) da Questão 4."""
    B_mat = np.array([
        [2.0, 1.0, 3.0],
        [0.0, 2.0, 2.0],
        [1.0, 0.0, 1.0],
    ], dtype=np.float64)
    y_B = np.array([5.0, 2.0, 1.0], dtype=np.float64)
    beta_0 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    return B_mat, y_B, beta_0


def get_sparse_synthetic_data(
    n_samples: int = 50,
    n_features: int = 10,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera o problema sintético esparso (N=50, d=10) da Questão Bônus."""
    rng = np.random.default_rng(seed)
    X_syn = rng.standard_normal((n_samples, n_features))
    w_true = np.array([3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    y_syn = X_syn @ w_true + noise_std * rng.standard_normal(n_samples)
    return X_syn, y_syn, w_true

