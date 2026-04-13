import time
from itertools import product
from .utils import evaluate_solution

def _evaluate_combination(index, values):
    """Evalua una combinacion y devuelve (indice, params, fitness)."""
    params = list(values)
    fitness = evaluate_solution(params)
    return index, params, fitness

def grid_search(patience=None, min_improvement=1e-4, use_early_stopping=True):
    # Definición de la rejilla (hay que ajustar para ~600 combinaciones)
    grid = {
        'n_estimators': [10, 100, 150, 200, 300],
        'max_depth': [2,10,20,30],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 10],
        'max_features': [0.3, 0.7],
        'bootstrap': [1],
        'criterion': [0, 1],
        'class_weight': [0, 1],
        'max_leaf_nodes': [100],
        'min_impurity_decrease': [0.0]
    }

    # Numero total de combinaciones sin materializar la rejilla completa en memoria
    n_comb = 1
    for values in grid.values():
        n_comb *= len(values)

    if use_early_stopping:
        if patience is None:
            patience = 10
        if patience <= 0:
            raise ValueError("patience debe ser mayor que 0")
        if min_improvement < 0:
            raise ValueError("min_improvement no puede ser negativo")
    
    print(f"\n--- Iniciando Grid Search ({n_comb} combinaciones) ---")

    best_fitness = -1
    best_params = None
    results_history = []
    start_time = time.time()

    no_significant_improvement = 0
    if use_early_stopping:
        print(f"Early stopping: patience={patience}, min_improvement={min_improvement}")
    else:
        print("Early stopping desactivado para Grid Search")

    for i, values in enumerate(product(*grid.values())):
        _, params, fitness = _evaluate_combination(i, values)
        results_history.append(fitness)

        prev_best = best_fitness
        significant_improvement = False

        if fitness > best_fitness:
            best_fitness = fitness
            best_params = params
            if (best_fitness - prev_best) > min_improvement:
                significant_improvement = True

        completed = i + 1
        print(
            f"Eval {completed}/{n_comb} (combinación {i+1}): "
            f"accuracy={fitness:.5f} | mejor={best_fitness:.5f}"
        )

        if significant_improvement:
            no_significant_improvement = 0
        else:
            no_significant_improvement += 1

        if use_early_stopping and completed >= patience and no_significant_improvement >= patience:
            print(
                "Parada anticipada activada: "
                f"{no_significant_improvement} evaluaciones sin mejora significativa "
                f"(umbral={min_improvement})."
            )
            break

    elapsed_time = time.time() - start_time
    return best_params, best_fitness, results_history, elapsed_time