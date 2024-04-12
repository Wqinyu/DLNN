# 三层神经网络图像分类器

本项目实现了一个基于三层神经网络的图像分类器，目标是在Fashion-MNIST数据集上进行训练和测试。

## 如何运行

1. 安装依赖:
   ```bash
   pip install -r requirements.txt

2. 在目录中创建data文件夹

3. 训练和测试模型

   ```bash
   python main.py
   ```

## 文件结构说明

- `cls.py`：包含`ThreeLayerNN`类，定义了三层神经网络的结构和方法。
- `data.py`：包含`MNISTDataHandler`类，用于处理Fashion-MNIST数据集的下载、加载和预处理。
- `main.py`：主运行文件，用于执行模型的训练和测试。
- `search_hyp.py`：用于进行超参数搜索的脚本。
- `search_draw.ipynb`：Jupyter notebook文件，用于绘制超参数搜索的结果。

## 数据集

本项目使用的数据集是Fashion-MNIST，可以通过`data.py`中的`MNISTDataHandler`类的`MNISTDataHandler`进行下载和加载。

## 训练和测试

- 训练过程中，模型的权重和偏置将保存在`best_model_weights`文件中。
- 测试过程中，将使用这些权重来评估模型在测试数据上的性能。

\## 超参数搜索结果 超参数搜索的结果保存在`hyperparameter_search_results.csv`文件中，可用于分析不同超参数对模型性能的影响。 

## 可视化结果

- `accuracy_curve.png`：训练和验证集上的准确率曲线。
- `loss_curve.png`：训练和验证集上的损失曲线。
- `weights_visualization.png`：模型第一层权重的部分可视化结果。
- `layers_weight_distribution.png`：网络各层权重的分布情况。
- `Validation Accuracy for Different Regularization and Hidden Sizes`：超参数网络搜索情况



