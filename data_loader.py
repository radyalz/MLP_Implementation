import os
import numpy as np
import struct
import tensorflow as tf

# تابع برای بارگذاری تصاویر MNIST از فایل
def load_mnist_images(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        print(f"Loading {filename}: magic={magic}, num={num}, rows={rows}, cols={cols}")  # Debugging statement
        images = np.fromfile(f, dtype=np.uint8)
        if images.size != num * rows * cols:
            raise ValueError(f"Size of the data does not match the expected size: {images.size} != {num * rows * cols}")
        images = images.reshape(num, rows * cols)
        return images / 255.0

# تابع برای بارگذاری لیبل‌های MNIST از فایل
def load_mnist_labels(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        print(f"Loading {filename}: magic={magic}, num={num}")  # Debugging statement
        labels = np.fromfile(f, dtype=np.uint8)
        if labels.size != num:
            raise ValueError(f"Size of the data does not match the expected size: {labels.size} != {num}")
        return labels

# تابع برای بارگذاری داده‌ها و تقسیم به مجموعه‌های آموزش، اعتبارسنجی و آزمون
def load_data():
    path = 'data'
    
    # بارگذاری داده‌های آموزشی
    x_train = load_mnist_images(os.path.join(path, 'train-images-idx3-ubyte'))
    y_train = load_mnist_labels(os.path.join(path, 'train-labels-idx1-ubyte'))
    
    # بارگذاری داده‌های آزمون
    x_test = load_mnist_images(os.path.join(path, 't10k-images-idx3-ubyte'))
    y_test = load_mnist_labels(os.path.join(path, 't10k-labels-idx1-ubyte'))
    
    # تقسیم داده‌های آموزشی به آموزش و اعتبارسنجی
    x_val = x_train[:10000]
    y_val = y_train[:10000]
    x_train = x_train[10000:]
    y_train = y_train[10000:]
    
    # تبدیل لیبل‌ها به فرمت one-hot
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return x_train, y_train, x_val, y_val, x_test, y_test
