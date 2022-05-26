from itertools import chain
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
classes = ['vertical line', 'horizontal line', 'two vertical lines', 'two horizontal lines']


def get_train_data():
    f = open('train', 'r')
    train_lines = [list(map(int, row.split())) for row in f.readlines()]
    train_data = [train_lines[i*7:(i+1)*7-1] for i in range(len(train_lines) // 7)]
    train_labels = list(chain([0] * 6, [1] * 6, [2] * 10, [3] * 10))
    return train_data, train_labels


def get_test_data():
    f = open('test', 'r')
    test_lines = [list(map(int, row.split())) for row in f.readlines()]
    test_data = [test_lines[i*7:(i+1)*7-1] for i in range(len(test_lines) // 7)]
    test_labels = [0, 1, 2, 3]
    return test_data, test_labels


def making_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(6, 6)),
        keras.layers.Dense(36, activation=tf.nn.relu),
        keras.layers.Dense(4, activation=tf.nn.softmax)
    ])
    return model


def compiling_and_training(model, data, labels):
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(data, labels, epochs=500)


def testing_accuracy(model, data, labels):
    test_loss, test_acc = model.evaluate(data, labels)
    return test_acc


def testing(model, data):
    predictions = model.predict(data)
    return predictions


def plot_image(i, predictions, test_labels, test_data):
    prediction, test_label, data = predictions[i], test_labels[i], test_data[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data, cmap=plt.cm.binary)
    predicted_label = np.argmax(prediction)
    if predicted_label == test_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(classes[predicted_label], 100 * np.max(prediction), classes[test_label]),
               color=color)


def plot_value_array(i, predictions, test_labels):
    prediction, test_label = predictions[i], test_labels[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plot = plt.bar(range(len(prediction)), prediction, color="grey")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction)
    plot[predicted_label].set_color('red')
    plot[test_label].set_color('green')


def main():
    train_data, train_labels = get_train_data()
    model = making_model()
    print('TRAINING:')
    compiling_and_training(model, train_data, train_labels)
    test_data, test_labels = get_test_data()
    print('\nTEST ACCURACY:')
    print(testing_accuracy(model, test_data, test_labels))
    predictions = testing(model, test_data)
    num_rows = 2
    num_cols = 2
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_data)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    plt.show()


main()
