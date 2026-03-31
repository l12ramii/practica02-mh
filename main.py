import pandas as pd 
import numpy as np 
 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import train_test_split 
 
 
# cargar dataset 
data = pd.read_csv("winequality-red.csv") 
 
# convertir problema a clasificación binaria 
data["quality"] = (data["quality"] >= 6).astype(int) 
 
X = data.drop("quality", axis=1) 
y = data["quality"] 
 
 
def evaluate_solution(params): 
 
    model = RandomForestClassifier( 
        n_estimators=int(params[0]), 
        max_depth=int(params[1]), 
        min_samples_split=int(params[2]), 
        min_samples_leaf=int(params[3]), 
        max_features=float(params[4]), 
        bootstrap=bool(params[5]), 
        criterion="gini" if params[6] == 0 else "entropy", 
        class_weight=None if params[7] == 0 else "balanced", 
        max_leaf_nodes=int(params[8]), 
        min_impurity_decrease=float(params[9]), 
        random_state=42 
    ) 
 
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy") 
 
    return scores.mean() 