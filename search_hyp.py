import os
from cls import ThreeLayerNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import seaborn as sns
from data import MNISTDataHandler

def hyperparameter_search(X_train, y_train, X_val, y_val):
    initial_lrs = [0.005, 0.01, 0.015]
    hidden_sizes = [512, 256, 1024]
    reg_lambdas = [0.1, 0.05, 0.001]
    results = []

    best_acc = -1
    best_params = {}

    for lr in initial_lrs:
        for hs in hidden_sizes:
            for reg in reg_lambdas:
                print(f"Training with lr={lr}, hidden_size={hs}, reg={reg}")
                model = ThreeLayerNN(784, hs, hs, 10, activation='relu')
                model.train(X_train, y_train, X_val, y_val, num_epochs=100, initial_lr=lr, reg_lambda=reg, early_stopping=True, patience=10, T_max=10)
                val_acc = model.best_val_acc
                print(f"Validation Accuracy: {val_acc:.4f}")
                results.append({'lr': lr, 'hidden_size': hs, 'reg': reg, 'val_acc': val_acc})
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_params = {'lr': lr, 'hidden_size': hs, 'reg': reg}


    results_df = pd.DataFrame(results)
    results_df.to_csv('hyperparameter_search_results.csv', index=False)

    print(f"Best hyperparameters: {best_params} with accuracy {best_acc:.4f}")
    return best_params
