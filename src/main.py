"""Script mestre de execução e validação determinística de todas as questões do projeto."""
from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.questao1 import run_questao1
from src.questao2 import run_questao2
from src.questao3 import run_questao3
from src.questao4 import run_questao4
from src.questao5 import run_questao5
from src.questao6 import run_questao6
from src.questao_bonus import run_questao_bonus


def main() -> None:
    np.random.seed(42)
    np.set_printoptions(precision=6, suppress=True)
    print("======================================================================")
    print("  SUITE DE VALIDAÇÃO — ÁLGEBRA LINEAR E OTIMIZAÇÃO PARA MACHINE LEARNING")
    print("======================================================================\n")
    run_questao1(save_figures=True)
    run_questao2(save_figures=True)
    run_questao3(save_figures=True)
    run_questao4(save_figures=True)
    run_questao5(save_figures=True)
    run_questao6(save_figures=True)
    run_questao_bonus(save_figures=True)
    print("Todas as validações analítico-numéricas e figuras foram geradas com sucesso!")


if __name__ == "__main__":
    main()

