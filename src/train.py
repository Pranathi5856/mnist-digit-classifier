# src/train.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.model import build_cnn, compile_model
import os

def load_and_preprocess():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)  # (N,28,28,1)
    x_test  = np.expand_dims(x_test, -1)
    return x_train, y_train, x_test, y_test

def main(out_dir="artifacts", epochs=8, batch_size=128, augment=False):
    os.makedirs(out_dir, exist_ok=True)
    x_train, y_train, x_test, y_test = load_and_preprocess()

    model = build_cnn((28,28,1), 10)
    compile_model(model)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir, "mnist_cnn.h5"),
                                           save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss")
    ]

    if augment:
        datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        datagen.fit(x_train)
        history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            steps_per_epoch=len(x_train)//batch_size,
                            callbacks=callbacks)
    else:
        history = model.fit(x_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(x_test, y_test),
                            callbacks=callbacks)

    # final evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

    # save full SavedModel
    model.save(os.path.join(out_dir, "mnist_cnn.keras"))
    return history

if __name__ == "__main__":
    main()
