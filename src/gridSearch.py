import time
from itertools import product
from .utils import evaluate_solution

def grid_search():
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

    # Generar todas las combinaciones posibles
    keys = list(grid.keys())
    combinations = list(product(*grid.values()))
    n_comb = len(combinations)
    
    print(f"\n--- Iniciando Grid Search ({n_comb} combinaciones) ---")
    
    best_fitness = -1
    best_params = None
    results_history = []
    start_time = time.time()

    for i, values in enumerate(combinations):
        # Mapear la combinación actual a la lista de 10 parámetros requerida
        params = list(values)
        
        fitness = evaluate_solution(params)
        results_history.append(fitness)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_params = params
            print(f"Combinación {i+1}/{n_comb}: Nuevo mejor accuracy -> {best_fitness:.5f}")

    elapsed_time = time.time() - start_time
    return best_params, best_fitness, results_history, elapsed_time