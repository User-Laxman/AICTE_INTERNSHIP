import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import sys

# Check TensorFlow version
print(f"TensorFlow Version: {tf.__version__}")

# Ensure GPU availability (optional)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Define colors
colors = {0: 'red', 1: 'green'}

# Create example function
def create_example(x, y):
    c = np.random.randint(0, 2)
    image = 0.5 * np.random.rand(28, 28, 3)
    image[:, :, c] += 0.5 * x / 255.
    return image, y, c

# Dataset Generator with tf.data.Dataset
def generate_data(x, y, batch_size=32):
    num_examples = len(y)
    def gen():
        while True:
            x_batch = np.zeros((batch_size, 28, 28, 3), dtype=np.float32)
            y_batch = np.zeros((batch_size,), dtype=np.float32)
            c_batch = np.zeros((batch_size,), dtype=np.float32)
            for i in range(batch_size):
                index = np.random.randint(0, num_examples)
                image, digit, color = create_example(x[index], y[index])
                x_batch[i] = image
                y_batch[i] = digit
                c_batch[i] = color
            yield x_batch, (y_batch, c_batch)  # Changed to tuple instead of list
    try:
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(batch_size, 28, 28, 3), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(batch_size,), dtype=tf.float32),
                    tf.TensorSpec(shape=(batch_size,), dtype=tf.float32)
                )
            )
        )
        return dataset
    except Exception as e:
        print(f"Error creating dataset: {e}")
        sys.exit(1)

# Create Model
num_filters = 32
input_ = tf.keras.layers.Input(shape=(28, 28, 3), name='input')
conv_1 = tf.keras.layers.Conv2D(num_filters, 3, name='conv_1')(input_)
act_1 = tf.keras.layers.Activation('relu', name='act_1')(conv_1)
pool_1 = tf.keras.layers.MaxPool2D(4, name='pool_1')(act_1)
flat_1 = tf.keras.layers.Flatten(name='flat_1')(pool_1)

conv_2 = tf.keras.layers.Conv2D(num_filters, 3, padding='same', name='conv_2')(act_1)
act_2 = tf.keras.layers.Activation('relu', name='act_2')(conv_2)
conv_3 = tf.keras.layers.Conv2D(num_filters, 3, padding='same', name='conv_3')(act_2)
add = tf.keras.layers.Add(name='add')([act_1, conv_3])
act_3 = tf.keras.layers.Activation('relu', name='act_3')(add)
pool_2 = tf.keras.layers.MaxPool2D(4, name='pool_2')(act_3)
flat_2 = tf.keras.layers.Flatten(name='flat_2')(pool_2)

digit = tf.keras.layers.Dense(10, activation='softmax', name='digit')(flat_2)
color = tf.keras.layers.Dense(1, activation='sigmoid', name='color')(flat_1)

model = tf.keras.models.Model(input_, [digit, color])

model.compile(
    loss={'digit': 'sparse_categorical_crossentropy', 'color': 'binary_crossentropy'},
    optimizer='adam',
    metrics={'digit': 'accuracy', 'color': 'accuracy'}
)

# Print model summary
model.summary()

# Save model architecture plot
tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True)
print("Model architecture saved as 'model_architecture.png'")

# Create datasets
train_gen = generate_data(x_train, y_train)
test_gen = generate_data(x_test, y_test)

# Clear logs directory if it exists
log_dir = './logs'
shutil.rmtree(log_dir, ignore_errors=True)
import time as t
min1 = int(t.strftime("%M"))
sec1 = int(t.strftime("%S"))
# Train the model
history = model.fit(
    train_gen,
    validation_data=test_gen,
    steps_per_epoch=200,
    validation_steps=100,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"Epoch {epoch+1}: "
                f"digit_acc: {logs.get('digit_accuracy'):.2f}, "
                f"color_acc: {logs.get('color_accuracy'):.2f}, "
                f"val_digit_acc: {logs.get('val_digit_accuracy'):.2f}, "
                f"val_color_acc: {logs.get('val_color_accuracy'):.2f}"
            )
        )
    ],
    verbose=1
)

# Save the trained model
model.save('mnist_color_model.h5')
print("Model saved as 'mnist_color_model.h5'")
min2 = int(t.strftime("%M"))
sec2 = int(t.strftime("%S")) 
min = min2 - min1
sec = sec2 - sec1
print("min : ", min if sec >=0 else min-1,"sec : ", sec if sec >= 0 else sec + 60)
# Final Prediction
def test_model(show=True, x=None, y=None, c=None, save_path=None):
    if x is None or y is None or c is None:
        x, (y, c) = next(iter(test_gen))  # Adjusted for tuple output
    preds = model.predict(x, verbose=0)
    pred_digit = np.argmax(preds[0][0])
    pred_color = int(preds[1][0] > 0.5)
    gt_digit = int(y[0])
    gt_color = int(c[0])

    plt.imshow(x[0])
    plt.plot()
    if show:
        
        print(f'GT: {gt_digit}, {colors[gt_color]}')
        print(f'Pr: {pred_digit}, {colors[pred_color]}')
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
    else:
        col = 'green' if gt_digit == pred_digit and gt_color == pred_color else 'red'
        plt.ylabel(f'GT: {gt_digit}, {colors[gt_color]}', color=col)
        plt.xlabel(f'Pr: {pred_digit}, {colors[pred_color]}', color=col)
        plt.xticks([])
        plt.yticks([])
        if save_path:
            plt.savefig(save_path)

# Single test prediction
test_model(show=True, save_path='single_prediction.png')

# Multiple predictions in a grid
plt.figure(figsize=(12, 12))
for i in range(16):
    plt.subplot(4, 4, i + 1)  
    test_model(show=False, save_path=f'grid_prediction_{i+1}.png')
plt.show()
plt.tight_layout()
plt.savefig('prediction_grid.png')
plt.close()
print("Prediction grid saved as 'prediction_grid.png'")

# Instructions for TensorBoard
print("Training complete. To view TensorBoard logs, run in terminal:")
print(f"tensorboard --logdir {log_dir}")