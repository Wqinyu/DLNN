import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

class ThreeLayerNN(object):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activation='relu', batch_size=64):
        # 使用He初始化方法
        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2. / hidden_size1)
        self.b1 = np.zeros((1, hidden_size1))
        
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size2)
        self.b2 = np.zeros((1, hidden_size2))
        
        self.W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2. / output_size)
        self.b3 = np.zeros((1, output_size))
        
        self.activation_name = activation
        self.batch_size = batch_size
        self.loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.train_purelose_history=[]
        self.val_purelose_history=[]
        self.best_params = {}
        self.best_val_acc = 0
        self.early_stopped_epoch = -1

    def save_params_npy(self, path='best_model_weights'):
        """以npy格式保存模型参数"""
        np.save(f'{path}_W1.npy', self.W1)
        np.save(f'{path}_b1.npy', self.b1)
        np.save(f'{path}_W2.npy', self.W2)
        np.save(f'{path}_b2.npy', self.b2)
        np.save(f'{path}_W3.npy', self.W3)
        np.save(f'{path}_b3.npy', self.b3)
        print(f"Model parameters saved in npy format at {path}.")

    def load_params_npy(self, path='best_model_weights'):
        """从npy文件加载模型参数"""
        self.W1 = np.load(f'{path}_W1.npy')
        self.b1 = np.load(f'{path}_b1.npy')
        self.W2 = np.load(f'{path}_W2.npy')
        self.b2 = np.load(f'{path}_b2.npy')
        self.W3 = np.load(f'{path}_W3.npy')
        self.b3 = np.load(f'{path}_b3.npy')

    def test(self, X_test, y_test):
        """评估模型在测试集上的性能"""
        test_acc = np.mean(self.predict(X_test) == y_test)
        print(f"Test Accuracy: {test_acc:.4f}")
        return test_acc

    def activation(self, z):
        if self.activation_name == 'relu':
            return np.maximum(0, z)
        else:
            return 1 / (1 + np.exp(-z))

    def activation_prime(self, z):
        if self.activation_name == 'relu':
            return (z > 0).astype(float)
        else:
            return self.activation(z) * (1 - self.activation(z))

    def forward(self, X):
        # 第一层前向传播
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        
        # 第二层前向传播
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = self.activation(self.z2)
        
        # 第三层（输出层）前向传播
        self.z3 = self.a2.dot(self.W3) + self.b3
        exp_scores = np.exp(self.z3)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def calculate_loss(self, X, y, reg_lambda=0.01):
        num_examples = X.shape[0]
        self.forward(X)
        correct_logprobs = -np.log(self.probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs) / num_examples
        reg_loss = 0.5 * reg_lambda * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)) + np.sum(np.square(self.W3)))
        return data_loss + reg_loss,data_loss

    def predict(self, X):
        self.forward(X)
        return np.argmax(self.probs, axis=1)
    
    def backprop(self, X, y, learning_rate, reg_lambda):
        num_examples = X.shape[0]
        delta4 = self.probs
        delta4[range(num_examples), y] -= 1
        delta4 /= num_examples
        
        # 更新W3和b3
        dW3 = np.dot(self.a2.T, delta4)
        db3 = np.sum(delta4, axis=0, keepdims=True)
        delta3 = np.dot(delta4, self.W3.T) * self.activation_prime(self.z2)
        
        # 更新W2和b2
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.W2.T) * self.activation_prime(self.z1)
        
        # 更新W1和b1
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)
        
        dW3 += reg_lambda * self.W3
        dW2 += reg_lambda * self.W2
        dW1 += reg_lambda * self.W1


        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

    

    def train(self, X_train, y_train, X_val, y_val, num_epochs=100, initial_lr=0.01, reg_lambda=0.01, early_stopping=True, patience=10, T_max=10):
        num_examples = X_train.shape[0]
        num_batches = max(num_examples // self.batch_size, 1)
        patience_counter = 0  # 初始化patience计数器
        lr_min = 0  # 学习率的最小值
        lr_max = initial_lr  # 初始学习率即为学习率的最大值

        for epoch in range(num_epochs):
            # 使用余弦退火调整学习率
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(epoch / T_max * np.pi))

            # 打乱
            shuffle_indices = np.random.permutation(num_examples)
            X_train_shuffled = X_train[shuffle_indices]
            y_train_shuffled = y_train[shuffle_indices]
        
            for i in range(0, num_examples, self.batch_size):
                # Mini-batch 
                end = i + self.batch_size
                X_batch = X_train_shuffled[i:end]
                y_batch = y_train_shuffled[i:end]
            
            self.forward(X_batch)
            self.backprop(X_batch, y_batch, lr, reg_lambda)

            # 计算epoch结束时的训练损失和准确率
            train_loss,train_pureloss = self.calculate_loss(X_train, y_train, reg_lambda)
            train_acc = np.mean(self.predict(X_train) == y_train)
    
            # 在验证集上评估模型
            val_loss,val_pureloss = self.calculate_loss(X_val, y_val, reg_lambda)
            val_acc = np.mean(self.predict(X_val) == y_val)

            # 保存损失和准确率历史
            self.loss_history.append(train_loss)
            self.train_purelose_history.append(train_pureloss)
            self.val_loss_history.append(val_loss)
            self.val_purelose_history.append(val_pureloss)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)

            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

            # 更新最佳模型参数
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_params_npy('best_model_weights')  # 保存最优权重
                self.best_params = {'W1': self.W1.copy(), 'b1': self.b1.copy(), 'W2': self.W2.copy(), 'b2': self.b2.copy(), 'W3': self.W3.copy(), 'b3': self.b3.copy()}  # 确保这里正确更新
                patience_counter = 0
            else:
                patience_counter += 1

            # 早停
            if early_stopping and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break



    def visualize_loss(self, filename_loss='loss_curve.png'):
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_purelose_history, label='Train')
        plt.plot(self.val_purelose_history, label='Validation')
        plt.title("Loss during training")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(filename_loss)
        plt.show()

    def visualize_accuracy(self, filename_acc='accuracy_curve.png'):
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_acc_history, label='Train')
        plt.plot(self.val_acc_history, label='Validation')
        plt.title("Accuracy during training")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(filename_acc)
        plt.show()

    def visualize_weights(self, filename_weights='weights_visualization.png'):
        if 'W1' not in self.best_params:
            print("Best model weights not available.")
            return

        fig, axes = plt.subplots(1, 10, figsize=(20, 2))
        for i, ax in enumerate(axes):
            if i < self.best_params['W1'].shape[1]:
                ax.imshow(self.best_params['W1'][:, i].reshape(28, 28), cmap='gray')
                ax.axis('off')
        plt.savefig(filename_weights)
        plt.show()


    def visualize_weight_distribution(self):
        """可视化权重分布"""
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.hist(self.W1.flatten(), bins=50)
        plt.title('Layer 1 Weight Distribution')
        plt.subplot(1, 3, 2)
        plt.hist(self.W2.flatten(), bins=50)
        plt.title('Layer 2 Weight Distribution')
        plt.subplot(1, 3, 3)
        plt.hist(self.W3.flatten(), bins=50)
        plt.title('Layer 3 Weight Distribution')
        plt.savefig("layers_weight_distribution.png")
        plt.show()    
