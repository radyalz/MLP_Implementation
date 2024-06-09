import subprocess
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from data_loader import load_data

# تابع برای نصب نیازمندی‌ها از فایل requirements.txt
# def install_requirements():
#     with open('requirements.txt', 'r') as f:
#         requirements = f.readlines()
    
#     for requirement in requirements:
#         package = requirement.strip()
#         if package:  # Ensure package is not an empty string
#             print(f"Installing package: {package}")  # Debugging statement
#             subprocess.check_call(['pip', 'install', package])

# تابع برای ایجاد مدل MLP
def create_mlp_model(layer_sizes, optimizer):
    model = Sequential()
    model.add(Flatten(input_shape=(28*28,)))
    for size in layer_sizes:
        model.add(Dense(size, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

# تابع برای ترسیم نتایج
def plot_results(history, title, save_dir):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(title + ' - Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{title}_accuracy.png"))  # Save plot

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title + ' - Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{title}_loss.png"))  # Save plot

    plt.show()

# مرحله ۱: بررسی تعداد لایه‌ها
def evaluate_layer_sizes(x_train, y_train, x_val, y_val, save_dir):
    layer_sizes_list = [[64], [64, 64, 64], [64, 64, 64, 64, 64]]
    histories = []

    for layer_sizes in layer_sizes_list:
        model = create_mlp_model(layer_sizes, Adam())
        history = model.fit(x_train, y_train, epochs=10, batch_size=16,
                            validation_data=(x_val, y_val), verbose=2)
        histories.append((layer_sizes, history))
        plot_results(history, f'Layers_{len(layer_sizes)}', save_dir)

    return histories

# مرحله ۲: بررسی تعداد نورون‌ها
def evaluate_neuron_counts(x_train, y_train, x_val, y_val, best_layers, save_dir):
    neuron_counts_list = [16, 64, 512]
    histories = []

    for count in neuron_counts_list:
        model = create_mlp_model([count] * len(best_layers), Adam())
        history = model.fit(x_train, y_train, epochs=10, batch_size=16,
                            validation_data=(x_val, y_val), verbose=2)
        histories.append((count, history))
        plot_results(history, f'Neurons_per_layer_{count}', save_dir)

    return histories

# مرحله ۳: بررسی اندازه Batch
def evaluate_batch_sizes(x_train, y_train, x_val, y_val, best_layers, best_neurons, save_dir):
    batch_sizes = [8, 16, 128]
    histories = []

    for batch_size in batch_sizes:
        model = create_mlp_model([best_neurons] * len(best_layers), Adam())
        history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size,
                            validation_data=(x_val, y_val), verbose=2)
        histories.append((batch_size, history))
        plot_results(history, f'Batch_size_{batch_size}', save_dir)

    return histories

# مرحله ۴: مقایسه بهینه‌سازها
def evaluate_optimizers(x_train, y_train, x_val, y_val, best_layers, best_neurons, save_dir):
    optimizers = {'SGD': SGD(), 'RMSprop': RMSprop()}
    histories = []

    for name, optimizer in optimizers.items():
        model = create_mlp_model([best_neurons] * len(best_layers), optimizer)
        history = model.fit(x_train, y_train, epochs=10, batch_size=16,
                            validation_data=(x_val, y_val), verbose=2)
        histories.append((name, history))
        plot_results(history, f'Optimizer_{name}', save_dir)

    return histories

# مرحله ۵: افزودن لایه Dropout
def evaluate_with_dropout(x_train, y_train, x_val, y_val, best_layers, best_neurons, save_dir):
    model = Sequential()
    model.add(Flatten(input_shape=(28*28,)))
    for size in [best_neurons] * len(best_layers):
        model.add(Dense(size, activation='relu'))
        model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=16,
                        validation_data=(x_val, y_val), verbose=2)
    plot_results(history, 'With_Dropout', save_dir)

    return history

# مرحله ۶: ارزیابی روی داده‌های آزمون
def evaluate_on_test_data(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}')

# تابع اصلی برای اجرای تمام مراحل
def main():
    # نصب نیازمندی‌ها
    # install_requirements()
    
    # دانلود مجموعه داده
    from setup import download_mnist
    download_mnist()
    
    # بارگذاری داده‌ها
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    
    # ایجاد پوشه برای ذخیره نتایج
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    # مرحله ۱
    layer_histories = evaluate_layer_sizes(x_train, y_train, x_val, y_val, save_dir)
    best_layers = [64, 64, 64]  # بر اساس تحلیل از مرحله ۱

    # مرحله ۲
    neuron_histories = evaluate_neuron_counts(x_train, y_train, x_val, y_val, best_layers, save_dir)
    best_neurons = 64  # بر اساس تحلیل از مرحله ۲

    # مرحله ۳
    batch_histories = evaluate_batch_sizes(x_train, y_train, x_val, y_val, best_layers, best_neurons, save_dir)

    # مرحله ۴
    optimizer_histories = evaluate_optimizers(x_train, y_train, x_val, y_val, best_layers, best_neurons, save_dir)

    # مرحله ۵
    dropout_history = evaluate_with_dropout(x_train, y_train, x_val, y_val, best_layers, best_neurons, save_dir)

    # مرحله ۶
    model = create_mlp_model([best_neurons] * len(best_layers), Adam())
    model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_val, y_val), verbose=2)
    evaluate_on_test_data(model, x_test, y_test)

if __name__ == "__main__":
    main()
