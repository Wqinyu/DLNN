import os
import requests
import gzip
import numpy as np

class MNISTDataHandler:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def download_extract_mnist(self, dataset_url, save_path):
        """下载并解压MNIST数据集"""
        response = requests.get(dataset_url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            with gzip.open(save_path, 'rb') as f_in:
                with open(save_path[:-3], 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(save_path)  # 删除压缩文件

    def download_data(self, download=False):
        """如果需要，则下载数据"""
        if download:
            base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
            files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
            for file in files:
                print(f'正在下载 {file}...')
                save_path = os.path.join(self.data_dir, file)
                self.download_extract_mnist(base_url + file, save_path[:-3])

    def load_mnist_images(self, filename):
        """加载MNIST图像数据"""
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            return data.reshape(-1, 784)

    def load_mnist_labels(self, filename):
        """加载MNIST标签数据"""
        with open(filename, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    def load_data(self):
        """加载完整的MNIST数据集"""
        X_train = self.load_mnist_images(os.path.join(self.data_dir, 'train-images-idx3-ub'))
        y_train = self.load_mnist_labels(os.path.join(self.data_dir, 'train-labels-idx1-ub'))
        X_test = self.load_mnist_images(os.path.join(self.data_dir, 't10k-images-idx3-ub'))
        y_test = self.load_mnist_labels(os.path.join(self.data_dir, 't10k-labels-idx1-ub'))

        print("training set(image)size:", X_train.shape)
        print("training set(label)size:", y_train.shape)
        print("test set(image)size:", X_test.shape)
        print("test set(label)size:", y_test.shape)

        return (X_train, y_train), (X_test, y_test)
