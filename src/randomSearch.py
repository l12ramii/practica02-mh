import numpy as np
from main import evaluate_solution 

def random_search(n_iter):
    best_accuracy = 0
    best_params = None
    results_history = []

    print(f"Iniciando Random Search ({n_iter} iteraciones)...")
    
    for i in range(n_iter):
        # Generar parámetros aleatorios según los rangos
        params = [
            np.random.randint(10, 301),       # n_estimators: 10-300 
            np.random.randint(2, 31),        # max_depth: 2-30 
            np.random.randint(2, 21),        # min_samples_split: 2-20 
            np.random.randint(1, 21),        # min_samples_leaf: 1-20 
            np.random.uniform(0.1, 1.0),     # max_features: 0.1-1.0 
            np.random.randint(0, 2),         # bootstrap: 0/1 
            np.random.randint(0, 2),         # criterion: 0=gini, 1=entropy 
            np.random.randint(0, 2),         # class_weight: 0=None, 1=balanced 
            np.random.randint(10, 201),      # max_leaf_nodes: 10-200 
            np.random.uniform(0, 0.1)        # min_impurity_decrease: 0-0.1 
        ]
        
        accuracy = evaluate_solution(params)
        results_history.append(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            print(f"Iteración {i+1}: Nuevo mejor accuracy encontrado -> {best_accuracy:.4f}")

    return best_params, best_accuracy, results_history