from config import *
from model.model import AUTOENCODER
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


class Train:
    def __init__(self) -> None:
        pass

    def train(self, Training_input, Autoencoder):
        """ """
        X_train, X_val, y_train, y_val = train_test_split(
            Training_input, Training_input, train_size=0.70
        )

        x_train = tf.one_hot(X_train, Num_symbols, dtype=tf.float32)
        y_train = tf.one_hot(y_train, Num_symbols, dtype=tf.float32)

        x_val = tf.one_hot(X_val, Num_symbols, dtype=tf.float32)
        y_val = tf.one_hot(y_val, Num_symbols, dtype=tf.float32)

        trained_model = Autoencoder.fit(
            x_train,
            y_train,
            epochs=45,
            batch_size=Batch_size,
            validation_data=(x_val, y_val),
        )
        return trained_model

    def plot(self, trained_model):
        """ """
        # summarize history for accuracy
        plt.plot(trained_model.history["accuracy"])
        plt.plot(trained_model.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["Train", "Validation"], loc="upper left")
        plt.show()

        # summarize history for loss
        plt.plot(trained_model.history["loss"])
        plt.plot(trained_model.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["Train", "Validation"], loc="upper left")
        plt.show()
