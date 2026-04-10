import numpy as np
import time
import copy
from .utils import evaluate_solution

def get_random_params():
    """Genera parámetros aleatorios respetando los rangos del enunciado[cite: 22]."""
    return [
        np.random.randint(10, 301),       # n_estimators
        np.random.randint(2, 31),        # max_depth
        np.random.randint(2, 21),        # min_samples_split
        np.random.randint(1, 21),        # min_samples_leaf
        np.random.uniform(0.1, 1.0),     # max_features
        np.random.randint(0, 2),         # bootstrap
        np.random.randint(0, 2),         # criterion
        np.random.randint(0, 2),         # class_weight
        np.random.randint(10, 201),      # max_leaf_nodes
        np.random.uniform(0, 0.1)        # min_impurity_decrease
    ]

def tournament_selection(population, fitnesses, k=3):
    """Selección por torneo."""
    idx = np.random.choice(len(population), k)
    best_idx = idx[np.argmax(np.array(fitnesses)[idx])]
    return population[best_idx]

def crossover_uniform(parent1, parent2):
    """Cruce uniforme gen a gen."""
    h1, h2 = [], []
    for i in range(len(parent1)):
        if np.random.rand() < 0.5:
            h1.append(parent1[i])
            h2.append(parent2[i])
        else:
            h1.append(parent2[i])
            h2.append(parent1[i])
    return h1, h2

def mutate(params, pm=0.1):
    """Mutación aleatoria de un gen."""
    new_params = copy.deepcopy(params)
    if np.random.rand() < pm:
        gene_to_mutate = np.random.randint(0, 10)
        random_indiv = get_random_params()
        new_params[gene_to_mutate] = random_indiv[gene_to_mutate]
    return new_params

def genetic_algorithm(mode="generational", pop_size=20, max_evals=10):
    """ Algoritmo Genético unificado (Generacional/Estacionario)."""
    start_time = time.time()
    results_history = []
    evals = 0
    
    print(f"\n--- Iniciando Algoritmo Genético ({mode}) ---")
    
    population = [get_random_params() for _ in range(pop_size)]
    fitnesses = []
    
    for i, ind in enumerate(population):
        fit = evaluate_solution(ind)
        fitnesses.append(fit)
        results_history.append(fit)
        evals += 1
        
        if evals == 1:
            best_fitness = fit
            best_params = ind
        elif fit > best_fitness:
            best_fitness = fit
            best_params = ind
            print(f"Eval {evals}/{max_evals}: Nuevo mejor accuracy -> {best_fitness:.5f}")

    while evals < max_evals:
        if mode == "generational":
            new_population = [best_params] 
            
            while len(new_population) < pop_size and evals < max_evals:
                p1 = tournament_selection(population, fitnesses)
                p2 = tournament_selection(population, fitnesses)
                h1, h2 = crossover_uniform(p1, p2)
                
                for h in [h1, h2]:
                    if len(new_population) < pop_size and evals < max_evals:
                        h_mutated = mutate(h)
                        fit = evaluate_solution(h_mutated)
                        new_population.append(h_mutated)
                        results_history.append(fit)
                        evals += 1

                        
                        if fit > best_fitness:
                            best_fitness = fit
                            best_params = h_mutated
                            print(f"Eval {evals}/{max_evals}: Nuevo mejor accuracy -> {best_fitness:.5f}")
            
            population = new_population
            fitnesses = [evaluate_solution(ind) for ind in population] 

        elif mode == "steady-state":
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            h1, h2 = crossover_uniform(p1, p2)
            
            for h in [h1, h2]:


                if evals < max_evals:
                    h_mutated = mutate(h)
                    fit = evaluate_solution(h_mutated)
                    results_history.append(fit)
                    evals += 1

                    
                    worst_idx = np.argmin(fitnesses)
                    if fit > fitnesses[worst_idx]:
                        population[worst_idx] = h_mutated
                        fitnesses[worst_idx] = fit
                    
                    if fit > best_fitness:
                        best_fitness = fit
                        best_params = h_mutated
                        print(f"Eval {evals}/{max_evals}: Nuevo mejor accuracy -> {best_fitness:.5f}")

    elapsed_time = time.time() - start_time
    return best_params, best_fitness, results_history, elapsed_time