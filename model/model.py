from config import *
from codebook.codeword_generation import *
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise, Dropout
from keras.models import Model
from keras import regularizers

# from tensorflow.keras.layers import BatchNormalization,Lambda
from keras.optimizers import Adam, SGD
from keras import backend as K
import math as m


class AUTOENCODER:
    def __init__(self, H_tr_to_RIS, H_RIS_to_rx, unique_codewords, codebook):
        self.H_tr_to_RIS = H_tr_to_RIS
        self.H_RIS_to_rx = H_RIS_to_rx
        self.unique_codewords = unique_codewords
        self.Num_of_codebook = len(unique_codewords)
        self.codebook = codebook

    def Tensor_multiplication_channel_tr_to_RIS(self, tensor):
        """
            Channel from the transmitter to RIS, htr, multiplies with\
            the output of the encoder, X.
        """
        htr = tf.dtypes.cast(tf.reshape(tensor[0], [1, RIS_elements]), tf.complex64)
        X = tensor[1]
        real = X[:, 0:1]
        imag = X[:, 1:]
        x = tf.complex(real, imag)

        cmplx_multiply = tf.math.multiply(x, htr)

        multi_real = tf.math.real(cmplx_multiply)
        multi_cmplx = tf.math.imag(cmplx_multiply)

        X_RIS = tf.dtypes.cast(
            tf.concat([multi_real, multi_cmplx], 1, name="concat"), tf.float32
        )
        return X_RIS

    def Tensor_multiplication_channel_h_rx(self, tensor):
        """
            Channel value from ris to receiver, h_rx, in multiplied \
            with the output of the RIS neural network module, Y_RIS. 
        """
        h_rx = tf.dtypes.cast(tf.reshape(tensor[0], [RIS_elements, 1]), tf.complex64)
        Y_RIS = tensor[1]
        y_real = Y_RIS[:, 0:RIS_elements]
        y_img = Y_RIS[:, RIS_elements:]
        y_ris = tf.complex(y_real, y_img)

        y_x = tf.linalg.matmul(y_ris, h_rx)
        y_x_real = tf.math.real(y_x)
        y_x_cmplx = tf.math.imag(y_x)

        y_x = tf.dtypes.cast(
            tf.concat([y_x_real, y_x_cmplx], 1, name="concat"), tf.float32
        )

        return y_x

    def Codeword_activation(self, tensor):
        """
            The codebook consists of different codewords, gets activated\
            when multiplied with the RIS output.  
        """
        print(self.codebook.shape, type(self.codebook))
        codebook_index = tensor[0]
        global count
        count = []
        count.append(codebook_index)

        X_RIS = tensor[1]
        x_real = X_RIS[:, 0:RIS_elements]
        x_img = X_RIS[:, RIS_elements:]
        x_ris = tf.complex(x_real, x_img)

        Index = codebook_index[0:]
        Codebook = tf.convert_to_tensor(
            self.codebook.reshape(self.Num_of_codebook, RIS_elements)
        )
        code = tf.dtypes.cast(tf.gather(Codebook, Index), tf.float32)
        rad_to_deg = m.pi / 180
        code_radian = tf.math.scalar_mul(rad_to_deg, code)

        phases = tf.complex(tf.math.cos(code_radian), tf.math.sin(code_radian))

        Y_RIS = tf.math.scalar_mul(Insertion_loss, phases)
        RIS = tf.math.multiply(x_ris, Y_RIS)

        r = tf.math.real(RIS)
        i = tf.math.imag(RIS)

        RIS = tf.dtypes.cast(tf.concat([r, i], 1, name="concat"), tf.float32)

        return RIS

    def Channel_ris_rx_concatenation(self, tensors):
        """ """
        RIS_h_rx = tensors[0]
        Channel_1 = tensors[1]
        RIS_h_rx = tf.reshape(
            tf.dtypes.cast(
                tf.concat(
                    [tf.math.real(RIS_h_rx), tf.math.imag(RIS_h_rx)], 0, name="concat"
                ),
                tf.float32,
            ),
            [1, 2 * RIS_elements],
        )
        RIS_h_rx = tf.repeat(RIS_h_rx, repeats=tf.shape(Channel_1)[0], axis=0)
        RIS_input = tf.dtypes.cast(
            tf.concat([Channel_1, RIS_h_rx], 1, name="concat"), tf.float32
        )
        return RIS_input

    def autoencoder_model(self):
        """ """
        # Num_of_codebook = len(codebook)
        RIS_input_size = int(RIS_elements * 4)

        """ Neural Network Model for the Autoencoder """
        input_signal = Input(shape=(Num_symbols,))

        """Encoder as a transmitter"""
        encoded_1 = Dense(Num_symbols, activation="relu")(input_signal)
        encoded_2 = Dense(Num_of_channel, activation="linear")(encoded_1)
        encoded = tf.keras.layers.Lambda(
            lambda x: np.sqrt(2) * K.l2_normalize(x, axis=1), name="Normalization"
        )(encoded_2)

        """ Channel multilication X_ris = encoded(shape: 2x1)*h_tr(shape:32x1)"""
        Channel_1 = tf.keras.layers.Lambda(
            self.Tensor_multiplication_channel_tr_to_RIS, name="Channel_tr"
        )([self.H_tr_to_RIS, encoded])
        RIS_input = tf.keras.layers.Lambda(
            self.Channel_ris_rx_concatenation, name="RIS_Input"
        )([self.H_RIS_to_rx, Channel_1])

        """RIS as a Neural Network """
        RIS_1 = Dense(RIS_input_size, activation="relu")(RIS_input)
        RIS_2 = Dense(RIS_input_size, activation="relu")(RIS_1)
        RIS_3 = Dense(RIS_input_size, activation="relu")(RIS_2)
        RIS_4 = Dense(RIS_input_size, activation="relu")(RIS_3)
        RIS_5 = Dense(RIS_input_size, activation="relu")(RIS_4)
        RIS = Dense(self.Num_of_codebook, activation="softmax")(RIS_5)

        """ Customized argmax function that will give the index value of the codebook vector"""
        codebook_index = tf.keras.layers.Lambda(
            lambda x: K.argmax(x), name="ArgMax_codebook"
        )(RIS)
        Y_RIS = tf.keras.layers.Lambda(
            self.Codeword_activation, name="Codebook_activation"
        )([codebook_index, Channel_1])

        """Channel Multilication y = Y_RIS*h_rx  """
        channel_h_rx = tf.keras.layers.Lambda(
            self.Tensor_multiplication_channel_h_rx, name="Channel_rx"
        )([self.H_RIS_to_rx, Y_RIS])
        Gaussian_noise = tf.keras.layers.GaussianNoise(
            m.sqrt(1 / 2 * Num_bits_per_channel * EbNo_train)
        )(channel_h_rx)

        """Decoder as a receiver """
        decoder_1 = Dense(Num_of_channel, activation="relu")(Gaussian_noise)
        Decoded = Dense(Num_symbols, activation="softmax")(decoder_1)

        Autoencoder = Model(input_signal, Decoded)
        Autoencoder.compile(
            optimizer=Adam(learning_rate=0.01),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        # Autoencoder.summary()

        return Autoencoder
