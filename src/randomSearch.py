import argparse
import numpy as np
import time

try:
    from .utils import evaluate_solution
except ImportError:
    from utils import evaluate_solution

DEFAULT_PATIENCE = 10
DEFAULT_MIN_IMPROVEMENT = 1e-4

def _generate_random_params():
    """Genera un conjunto de hiperparametros aleatorios dentro de los rangos definidos."""
    return [
        np.random.randint(10, 301),       # n_estimators: 10-300
        np.random.randint(2, 31),         # max_depth: 2-30
        np.random.randint(2, 21),         # min_samples_split: 2-20
        np.random.randint(1, 21),         # min_samples_leaf: 1-20
        np.random.uniform(0.1, 1.0),      # max_features: 0.1-1.0
        np.random.randint(0, 2),          # bootstrap: 0/1
        np.random.randint(0, 2),          # criterion: 0=gini, 1=entropy
        np.random.randint(0, 2),          # class_weight: 0=None, 1=balanced
        np.random.randint(10, 201),       # max_leaf_nodes: 10-200
        np.random.uniform(0, 0.1)         # min_impurity_decrease: 0-0.1
    ]

def _evaluate_iteration(index, params):
    """Evalua una iteracion de random search y devuelve (indice, params, accuracy)."""
    accuracy = evaluate_solution(params)
    return index, params, accuracy

def random_search(n_iter, patience=None, min_improvement=DEFAULT_MIN_IMPROVEMENT):
    if n_iter <= 0:
        raise ValueError("n_iter debe ser mayor que 0")
    if patience is None:
        patience = DEFAULT_PATIENCE
    if patience <= 0:
        raise ValueError("patience debe ser mayor que 0")
    if min_improvement < 0:
        raise ValueError("min_improvement no puede ser negativo")

    best_accuracy = 0
    best_params = None
    results_history = []

    print(f"Iniciando Random Search ({n_iter} iteraciones)...")

    start_time = time.time()
    no_significant_improvement = 0
    print(f"Early stopping: patience={patience}, min_improvement={min_improvement}")
    for i in range(n_iter):
        params = _generate_random_params()
        _, params, accuracy = _evaluate_iteration(i, params)
        results_history.append(accuracy)

        prev_best = best_accuracy
        significant_improvement = False

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            if (best_accuracy - prev_best) > min_improvement:
                significant_improvement = True

        print(
            f"Eval {i+1}/{n_iter} (iteración {i+1}): "
            f"accuracy={accuracy:.5f} | mejor={best_accuracy:.5f}"
        )

        if significant_improvement:
            no_significant_improvement = 0
        else:
            no_significant_improvement += 1

        if (i + 1) >= patience and no_significant_improvement >= patience:
            print(
                "Parada anticipada activada: "
                f"{no_significant_improvement} evaluaciones sin mejora significativa "
                f"(umbral={min_improvement})."
            )
            break

    stop_time = time.time()
    return best_params, best_accuracy, results_history, stop_time - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta Random Search")
    parser.add_argument("--n-iter", type=int, default=10, help="Número de iteraciones")
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Iteraciones sin mejora significativa para parar",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=DEFAULT_MIN_IMPROVEMENT,
        help="Umbral mínimo de mejora considerado significativo",
    )
    args = parser.parse_args()

    best_params, best_accuracy, history, duration = random_search(
        n_iter=args.n_iter,
        patience=args.patience,
        min_improvement=args.min_improvement
    )

    print("\n--- Resultado ejecución directa ---")
    print(f"Mejor accuracy: {best_accuracy:.5f}")
    print(f"Tiempo total: {duration:.2f}s")
    print(f"Número de evaluaciones: {len(history)}")
    print(f"Mejores parámetros: {best_params}")