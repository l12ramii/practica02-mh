from src.randomSearch import random_search
from src.gridSearch import grid_search
from src.geneticAlgorithm import genetic_algorithm


def ask_early_stopping_config(total_iterations=None, default_min_improvement=1e-4):
    """Solicita configuracion de parada anticipada.

    Si patience se deja vacio, usa 10 iteraciones.
    """
    auto_patience = 10
    patience_hint = f"auto: {auto_patience}"

    patience_input = input(f"Patience (iteraciones sin mejora) [{patience_hint}]: ").strip()
    min_imp_input = input(f"Umbral minimo de mejora [{default_min_improvement}]: ").strip()

    patience = auto_patience if patience_input == "" else int(patience_input)
    min_improvement = default_min_improvement if min_imp_input == "" else float(min_imp_input)

    if patience is not None and patience <= 0:
        raise ValueError("Patience debe ser mayor que 0")
    if min_improvement < 0:
        raise ValueError("El umbral minimo de mejora no puede ser negativo")

    return patience, min_improvement


def ask_max_evals(default_max_evals=600):
    """Solicita el numero maximo de evaluaciones para el algoritmo genetico."""
    max_evals_input = input(f"Número máximo de evaluaciones [{default_max_evals}]: ").strip()
    max_evals = default_max_evals if max_evals_input == "" else int(max_evals_input)
    if max_evals <= 0:
        raise ValueError("El número de evaluaciones debe ser mayor que 0")
    return max_evals

if __name__ == "__main__":
    while True:
        print("\n=== Práctica 2: Metaheurísticas ===")
        print("1. Ejecutar Random Search (Especificar iteraciones)")
        print("2. Grid Search (Rejilla predefinida)")
        print("3. Algoritmo Genético (Generacional)")
        print("4. Algoritmo Genético (Estacionario)")
        print("5. Salir")
        
        opcion = input("\nSeleccione una opción: ")

        if opcion == "1":
            try:
                n = int(input("Introduce el número de iteraciones: "))
                patience, min_improvement = ask_early_stopping_config(total_iterations=n)
                best_params, best_accuracy, results_history, duration = random_search(
                    n_iter=n,
                    patience=patience,
                    min_improvement=min_improvement
                )
                print(f"\nResultado Random Search: {best_accuracy:.4f} en {duration:.2f}s")
            except ValueError:
                print("Error: Por favor, introduce un número válido.")
        elif opcion == "2":
            try:
                patience, min_improvement = ask_early_stopping_config()
                best_p, best_acc, results_history, duration = grid_search(
                    patience=patience,
                    min_improvement=min_improvement
                )
                print(f"\nResultado Grid Search: {best_acc:.4f} en {duration:.2f}s")
            except ValueError:
                print("Error: Parámetros de parada anticipada no válidos.")
        elif opcion == "3":
            try:
                max_evals = ask_max_evals()
                patience, min_improvement = ask_early_stopping_config(total_iterations=max_evals)
                best_p, best_acc, history, duration = genetic_algorithm(
                    mode="generational",
                    max_evals=max_evals,
                    patience=patience,
                    min_improvement=min_improvement
                )
                print(f"\nResultado AG Generacional: {best_acc:.4f} en {duration:.2f}s")
            except ValueError:
                print("Error: número de evaluaciones no válido.")
        elif opcion == "4":
            try:
                max_evals = ask_max_evals()
                patience, min_improvement = ask_early_stopping_config(total_iterations=max_evals)
                best_p, best_acc, history, duration = genetic_algorithm(
                    mode="steady-state",
                    max_evals=max_evals,
                    patience=patience,
                    min_improvement=min_improvement
                )
                print(f"\nResultado AG Estacionario: {best_acc:.4f} en {duration:.2f}s")
            except ValueError:
                print("Error: número de evaluaciones no válido.")
        elif opcion == "5":
            print("\nSaliendo...")
            break
        else:
            print("Opción no válida.")
