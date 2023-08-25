from config import *
import math as m
import tensorflow as tf


class CHANNELS:
    def __init__(self) -> None:
        pass

    def Channel_values_generation(self, Eleveation, Azimuth):

        """
            Generates the channel values for both, transmitter\
            to RIS and RIS to receiver.    
        """

        PHI = (
            RIS_element_gap
            * m.sin(m.radians(Azimuth))
            * m.cos(m.radians(Eleveation))
            / Wavelength
        )
        PSI = RIS_element_gap * m.sin(m.radians(Eleveation)) / Wavelength

        angle_1 = 2 * m.pi * PHI
        angle_2 = 2 * m.pi * PSI

        arr_1 = []
        arr_2 = []

        for i in range(RIS_rows):
            ang = angle_1 * i
            arr_1.append(tf.math.exp(complex(0, ang)))
        Matrix_1 = tf.reshape(tf.convert_to_tensor(arr_1), [RIS_rows, 1])

        for i in range(RIS_cols):
            ang = angle_2 * i
            arr_2.append(tf.math.exp(complex(0, ang)))
        Matrix_2 = tf.reshape(tf.convert_to_tensor(arr_2), [RIS_cols, 1])

        matrix1_kronecler_matrix2 = tf.linalg.LinearOperatorKronecker(
            [
                tf.linalg.LinearOperatorFullMatrix(Matrix_1),
                tf.linalg.LinearOperatorFullMatrix(Matrix_2),
            ]
        )
        alpha = tf.convert_to_tensor(
            np.multiply((1 / np.sqrt(2)), np.vectorize(complex)(1, 1))
        )

        channel_values = tf.multiply(alpha, matrix1_kronecler_matrix2.to_dense())

        return channel_values
