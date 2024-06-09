# import os
# import tensorflow as tf
# # دانلود کننده دیتاست
# def download_mnist():
#     dataset_dir = "data"
#     os.makedirs(dataset_dir, exist_ok=True)
    
#     # بارگذاری و دانلود دیتاست با استفاده از تنسرفلو
#     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#     # فایل ها را در مکان مشخصی که تایین کردم ذخیره میکنم
#     train_images_path = os.path.join(dataset_dir, 'train-images-idx3-ubyte')
#     train_labels_path = os.path.join(dataset_dir, 'train-labels-idx1-ubyte')
#     test_images_path = os.path.join(dataset_dir, 't10k-images-idx3-ubyte')
#     test_labels_path = os.path.join(dataset_dir, 't10k-labels-idx1-ubyte')
    
#     # ذخیره داده های آموزشی 
#     x_train.tofile(train_images_path)
#     y_train.tofile(train_labels_path)
    
#     # ذخیره داده های تست 
#     x_test.tofile(test_images_path)
#     y_test.tofile(test_labels_path)

# if __name__ == "__main__":
#     download_mnist()

import os
import numpy as np
import struct
import tensorflow as tf

def save_mnist_images(filename, images):
    with open(filename, 'wb') as f:
        magic = 2051
        num_images = images.shape[0]
        rows = images.shape[1]
        cols = images.shape[2]
        f.write(struct.pack('>IIII', magic, num_images, rows, cols))
        images.tofile(f)

def save_mnist_labels(filename, labels):
    with open(filename, 'wb') as f:
        magic = 2049
        num_labels = labels.shape[0]
        f.write(struct.pack('>II', magic, num_labels))
        labels.tofile(f)

def download_mnist():
    dataset_dir = "data"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Load the MNIST dataset using TensorFlow
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Save training data
    save_mnist_images(os.path.join(dataset_dir, 'train-images-idx3-ubyte'), x_train)
    save_mnist_labels(os.path.join(dataset_dir, 'train-labels-idx1-ubyte'), y_train)
    
    # Save test data
    save_mnist_images(os.path.join(dataset_dir, 't10k-images-idx3-ubyte'), x_test)
    save_mnist_labels(os.path.join(dataset_dir, 't10k-labels-idx1-ubyte'), y_test)

if __name__ == "__main__":
    download_mnist()
