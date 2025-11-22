# src/evaluate.py
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from src.model import build_cnn, compile_model

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1)
    return x_test, y_test

def plot_confusion(cm, classes):
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def main(model_path="artifacts/mnist_cnn.h5"):
    x_test, y_test = load_data()
    model = tf.keras.models.load_model(model_path)
    preds = model.predict(x_test).argmax(axis=1)
    cm = confusion_matrix(y_test, preds)
    plot_confusion(cm, list(range(10)))

if __name__ == "__main__":
    main()
