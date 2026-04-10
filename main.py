from src.randomSearch import random_search
from src.utils import plot_results
from src.gridSearch import grid_search
from src.geneticAlgorithm import genetic_algorithm

if __name__ == "__main__":
    while True:
        print("\n=== Práctica 2: Metaheurísticas ===")
        print("1. Ejecutar Random Search (600 iteraciones por defecto)")
        print("2. Ejecutar Random Search (Especificar iteraciones)")
        print("3. Grid Search (Rejilla predefinida)")
        print("4. Algoritmo Genético (Generacional)")
        print("5. Algoritmo Genético (Estacionario)")
        print("6. Salir")
        
        opcion = input("\nSeleccione una opción: ")

        if opcion == "1":
            best_params, best_accuracy, results_history, elapsed_time = random_search(n_iter=600)
            print(f"\nResultado: {best_accuracy:.4f} en {elapsed_time:.2f}s")
            plot_results(history=results_history, n_iter=600, alg="Random Search")
        elif opcion == "2":
            try:
                n = int(input("Introduce el número de iteraciones: "))
                best_params, best_accuracy, results_history, time = random_search(n_iter=n)
            except ValueError:
                print("Error: Por favor, introduce un número válido.")
        elif opcion == "3":
            best_p, best_acc, results_history, duration = grid_search()
            print(f"\nResultado Grid Search: {best_acc:.4f} en {duration:.2f}s")
            plot_results(history= results_history, n_iter=len(results_history), alg="Grid Search")
        elif opcion == "4":
            best_p, best_acc, history, duration = genetic_algorithm(mode="generational")
            print(f"\nResultado AG Generacional: {best_acc:.4f} en {duration:.2f}s")
            plot_results(history, len(history), "AG Generacional")
        elif opcion == "5":
            best_p, best_acc, history, duration = genetic_algorithm(mode="steady-state")
            print(f"\nResultado AG Estacionario: {best_acc:.4f} en {duration:.2f}s")
            plot_results(history, len(history), "AG Estacionario")
        elif opcion == "6":
            print("\nSaliendo...")
            break
        else:
            print("Opción no válida.")
