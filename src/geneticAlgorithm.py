import numpy as np
import time
from .utils import evaluate_solution

# Limites por gen: [min, max] para los 10 hiperparametros.
LIMITS = np.array([
    [10, 300],    # n_estimators (int)
    [2, 30],      # max_depth (int)
    [2, 20],      # min_samples_split (int)
    [1, 20],      # min_samples_leaf (int)
    [0.1, 1.0],   # max_features (float)
    [0, 1],       # bootstrap (binario)
    [0, 1],       # criterion (binario)
    [0, 1],       # class_weight (binario)
    [10, 200],    # max_leaf_nodes (int)
    [0.0, 0.1]    # min_impurity_decrease (float)
], dtype=float)

INTEGER_GENES = {0, 1, 2, 3, 8}
BINARY_GENES = {5, 6, 7}

def get_random_params():
    """Genera parámetros aleatorios respetando los rangos del enunciado[cite: 22]."""
    return [
        np.random.randint(10, 301),      # n_estimators
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

def roulette_selection(population, fitnesses):
    """Seleccion por ruleta estocastica proporcional al fitness."""
    fitness_array = np.array(fitnesses, dtype=float)

    # Si hay valores negativos, desplazar para mantener probabilidades validas.
    min_fit = np.min(fitness_array)
    if min_fit < 0:
        fitness_array = fitness_array - min_fit

    total_fitness = np.sum(fitness_array)

    # Si todos tienen fitness 0, seleccionar de forma uniforme.
    if total_fitness <= 0:
        selected_idx = np.random.randint(0, len(population))
        return population[selected_idx]

    probabilities = fitness_array / total_fitness
    selected_idx = np.random.choice(len(population), p=probabilities)
    return population[selected_idx]

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

def mutate(params, pm=0.3, sigma=0.2):
    """Mutacion gaussiana por gen con recorte a limites del dominio."""
    mutado = np.array(params, dtype=float, copy=True)

    for i in range(len(mutado)):
        if np.random.rand() < pm:
            rango = LIMITS[i, 1] - LIMITS[i, 0]
            ruido = np.random.normal(0, sigma * rango)
            mutado[i] += ruido
            mutado[i] = np.clip(mutado[i], LIMITS[i, 0], LIMITS[i, 1])

    # Restaurar tipos esperados por cada gen.
    for i in range(len(mutado)):
        if i in BINARY_GENES:
            mutado[i] = int(np.clip(np.round(mutado[i]), 0, 1))
        elif i in INTEGER_GENES:
            mutado[i] = int(np.round(mutado[i]))
        else:
            mutado[i] = float(mutado[i])

    return mutado.tolist()

def genetic_algorithm(mode="generational", pop_size=20, max_evals=10, patience=None, min_improvement=1e-4):
    """ Algoritmo Genético unificado (Generacional/Estacionario)."""
    if max_evals <= 0:
        raise ValueError("max_evals debe ser mayor que 0")
    if patience is None:
        patience = 10
    if patience <= 0:
        raise ValueError("patience debe ser mayor que 0")
    if min_improvement < 0:
        raise ValueError("min_improvement no puede ser negativo")

    start_time = time.time()
    results_history = []
    evals = 0
    no_significant_improvement = 0
    early_stopped = False
    
    print(f"\n--- Iniciando Algoritmo Genético ({mode}) ---")
    print(f"Early stopping: patience={patience}, min_improvement={min_improvement}")
    population = [get_random_params() for _ in range(pop_size)]
    fitnesses = []

    for ind in population:
        if evals >= max_evals:
            break

        fit = evaluate_solution(ind)
        fitnesses.append(fit)
        results_history.append(fit)
        evals += 1

        if evals == 1:
            best_fitness = fit
            best_params = ind
            significant_improvement = True
        elif fit > best_fitness:
            prev_best = best_fitness
            best_fitness = fit
            best_params = ind
            significant_improvement = (best_fitness - prev_best) > min_improvement
        else:
            significant_improvement = False

        if significant_improvement:
            no_significant_improvement = 0
        else:
            no_significant_improvement += 1

        print(
            f"Eval {evals}/{max_evals}: accuracy={fit:.5f} | "
            f"mejor={best_fitness:.5f}"
        )

        if evals >= patience and no_significant_improvement >= patience:
            early_stopped = True
            print(
                "Parada anticipada activada: "
                f"{no_significant_improvement} evaluaciones sin mejora significativa "
                f"(umbral={min_improvement})."
            )
            break

    while evals < max_evals and not early_stopped:
        if mode == "generational":
            new_population = [best_params]
            new_fitnesses = [best_fitness]

            while len(new_population) < pop_size and evals < max_evals:
                p1 = roulette_selection(population, fitnesses)
                p2 = roulette_selection(population, fitnesses)
                h1, h2 = crossover_uniform(p1, p2)

                for h in [h1, h2]:
                    if len(new_population) < pop_size and evals < max_evals:
                        h_mutated = mutate(h)
                        fit = evaluate_solution(h_mutated)
                        new_population.append(h_mutated)
                        new_fitnesses.append(fit)
                        results_history.append(fit)
                        evals += 1

                        significant_improvement = False
                        if fit > best_fitness:
                            prev_best = best_fitness
                            best_fitness = fit
                            best_params = h_mutated
                            significant_improvement = (best_fitness - prev_best) > min_improvement

                        if significant_improvement:
                            no_significant_improvement = 0
                        else:
                            no_significant_improvement += 1

                        print(
                            f"Eval {evals}/{max_evals}: accuracy={fit:.5f} | "
                            f"mejor={best_fitness:.5f}"
                        )

                        if evals >= patience and no_significant_improvement >= patience:
                            early_stopped = True
                            print(
                                "Parada anticipada activada: "
                                f"{no_significant_improvement} evaluaciones sin mejora significativa "
                                f"(umbral={min_improvement})."
                            )
                            break

                if early_stopped:
                    break

            population = new_population
            fitnesses = new_fitnesses

        elif mode == "steady-state":
            p1 = roulette_selection(population, fitnesses)
            p2 = roulette_selection(population, fitnesses)
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

                    significant_improvement = False
                    if fit > best_fitness:
                        prev_best = best_fitness
                        best_fitness = fit
                        best_params = h_mutated
                        significant_improvement = (best_fitness - prev_best) > min_improvement

                    if significant_improvement:
                        no_significant_improvement = 0
                    else:
                        no_significant_improvement += 1

                    print(
                        f"Eval {evals}/{max_evals}: accuracy={fit:.5f} | "
                        f"mejor={best_fitness:.5f}"
                    )

                    if evals >= patience and no_significant_improvement >= patience:
                        early_stopped = True
                        print(
                            "Parada anticipada activada: "
                            f"{no_significant_improvement} evaluaciones sin mejora significativa "
                            f"(umbral={min_improvement})."
                        )
                        break

            if early_stopped:
                break

    elapsed_time = time.time() - start_time
    return best_params, best_fitness, results_history, elapsed_time