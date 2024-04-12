import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data import MNISTDataHandler
from cls import ThreeLayerNN
import numpy as np
from search_hyp import hyperparameter_search


if __name__ == "__main__":

    data_handler = MNISTDataHandler()

    data_handler.download_data(download=True)

    (X_train, y_train), (X_test, y_test) = data_handler.load_data()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 数据预处理
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    X_val = X_val.reshape(X_val.shape[0], -1) / 255.0

    best_parm=hyperparameter_search(X_train, y_train, X_val, y_val)
    # print(best_parm)

    model = ThreeLayerNN(784, best_parm['hidden_size'], best_parm['hidden_size'], 10, activation='relu')

    model.train(X_train, y_train, X_val, y_val, num_epochs=200, initial_lr=best_parm['lr'], reg_lambda=best_parm['reg'], early_stopping=False, patience=15)
    
    # 可视化训练和验证损失及准确率
    model.visualize_loss()
    model.visualize_accuracy()

    # 保存模型权重
    model.save_params_npy('best_model_weights')

    # 加载模型权重
    model.load_params_npy('best_model_weights')

    model.test(X_test, y_test)

    # 可视化学习到的权重
    model.visualize_weights()

    # 可视化权重分布
    model.visualize_weight_distribution()
