import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score 
import matplotlib.pyplot as plt

# Obtener ruta del directorio del proyecto
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_dir, "data", "winequality-red.csv")

# Cargar dataset
data = pd.read_csv(data_path, sep=",") 

# Convertir problema a clasificación binaria (bueno >= 6)
data["quality"] = (data["quality"] >= 6).astype(int) 

X = data.drop("quality", axis=1) 
y = data["quality"] 

def evaluate_solution(params): 
    model = RandomForestClassifier( 
        n_estimators=int(params[0]),           # Rango: 10-300 
        max_depth=int(params[1]),              # Rango: 2-30 
        min_samples_split=int(params[2]),      # Rango: 2-20 
        min_samples_leaf=int(params[3]),       # Rango: 1-20 
        max_features=float(params[4]),         # Rango: 0.1-1.0 
        bootstrap=bool(params[5]),             # Binario: 0/1 
        criterion="gini" if params[6] == 0 else "entropy", # 0=gini, 1=entropy
        class_weight=None if params[7] == 0 else "balanced", # 0=None, 1=balanced
        max_leaf_nodes=int(params[8]),         # Rango: 10-200 
        min_impurity_decrease=float(params[9]),# Rango: 0-0.1 
        random_state=42                        # Estado fijo
    ) 

    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    return scores.mean() 

import matplotlib.pyplot as plt
import numpy as np

# Genera y guarda la gráfica de accuracy vs iteración
def plot_results(history, n_iter, alg):
    plt.figure(figsize=(10, 6))
    
    # Obtener media y desviación típica
    mean_acc = np.mean(history)
    std_acc = np.std(history, ddof=1)
    
    running_max = np.maximum.accumulate(history)
    last_best = running_max[-1]
    
    label_stats = f'Accuracy ($\\bar{{x}}$={mean_acc:.4f}, $s$={std_acc:.4f})'
    
    # Accuracy de cada iteración 
    plt.plot(range(1, n_iter + 1), history, label=label_stats, alpha=0.4, color='blue')
    
    # Mejor accuracy hasta el momento
    plt.plot(range(1, n_iter + 1), running_max, label='Mejor Accuracy', color='red', linewidth=2)
    
    # Anotar el mejor valor de la accuracy
    plt.annotate(f'{last_best:.4f}', 
                 xy=(n_iter, last_best), 
                 xytext=(5, 5), 
                 textcoords='offset points',
                 fontsize=10,
                 fontweight='bold',
                 color='red')

    plt.xlabel('Iteración')
    plt.ylabel('Accuracy')
    plt.title(f'Evolución de la Accuracy ({alg} - {n_iter} iteraciones)')
    plt.legend()
    plt.grid(True)
    
    # Guardar imagen
    filename = f"random_search_{n_iter}_results.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
