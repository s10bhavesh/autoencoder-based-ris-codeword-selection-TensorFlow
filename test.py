from config import *
from model.model import AUTOENCODER
import numpy as np
from keras.layers import Input, Dense, GaussianNoise, Dropout
from keras.models import Model
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt


class Test:
    def __init__(self) -> None:
        pass

    def frange(self, x, y, jump):
        while x < y:
            yield x
            x += jump

    def Test_model(self, Autoencoder):
        RIS_input_size = int(RIS_elements * 4)
        input_signal = Input(shape=(Num_symbols,))

        encoder_1 = Autoencoder.layers[0](input_signal)
        encoder_2 = Autoencoder.layers[1](encoder_1)
        encoder_3 = Autoencoder.layers[2](encoder_2)
        encoder = Model(encoder_1, encoder_3)
        encoder.summary()

        ris_input = Input(shape=(RIS_input_size))
        ris_1 = Autoencoder.layers[-12](ris_input)
        ris_2 = Autoencoder.layers[-11](ris_1)
        ris_3 = Autoencoder.layers[-10](ris_2)
        ris_4 = Autoencoder.layers[-9](ris_3)
        ris_5 = Autoencoder.layers[-8](ris_4)
        ris_6 = Autoencoder.layers[-7](ris_5)
        ris = Model(ris_input, ris_6)
        ris.summary()

        decoded_input = Input(shape=(2,))
        decoder_1 = Autoencoder.layers[-2](decoded_input)
        decoder_2 = Autoencoder.layers[-1](decoder_1)
        Decoder = Model(decoded_input, decoder_2)
        Decoder.summary()
        return encoder, ris, Decoder

    def Test(self, H_tr_to_RIS, H_RIS_to_rx, Testing_input, Autoencoder, model):
        encoder, ris, Decoder = self.Test_model(Autoencoder)

        Test_data = tf.one_hot(Testing_input, Num_symbols, dtype=tf.int64)

        def constellation_plot(self):
            # for plotting learned consteallation diagram
            scatter_plot = []
            for i in range(0, Num_symbols):
                print(i)
            temp = np.zeros(Num_symbols)
            temp[i] = 1
            print(np.expand_dims(temp, axis=0))
            scatter_plot.append(encoder.predict(np.expand_dims(temp, axis=0)))
            scatter_plot = np.array(scatter_plot)
            print(scatter_plot.shape)
            print(scatter_plot)

            scatter_plot = scatter_plot.reshape(Num_symbols, 2, 1)
            plt.scatter(scatter_plot[:, 0], scatter_plot[:, 1])
            plt.axis((-2.5, 2.5, -2.5, 2.5))
            plt.grid()
            plt.show()

        EbNodB_range = list(self.frange(-15, 20, 1))
        No_iter = 1
        BER_list = []

        print(f"{i}nd iteration\n")
        ber = [None] * len(EbNodB_range)
        for n in range(0, len(EbNodB_range)):
            EbNo = 10.0 ** (EbNodB_range[n] / 10.0)
            noise_std = np.sqrt(1 / (2 * Num_bits_per_channel * EbNo))
            noise_mean = 0
            no_errors = 0
            nn = Test_sample_size
            noise = noise_std * np.random.randn(nn, Num_of_channel)

            encoded_signal = encoder.predict(Test_data)
            Channel_1 = model.Tensor_multiplication_channel_tr_to_RIS(
                [H_tr_to_RIS, encoded_signal]
            )
            RIS_input = model.Channel_ris_rx_concatenation([H_RIS_to_rx, Channel_1])
            ris_signal = ris.predict(RIS_input)
            codebook_index = K.argmax(ris_signal)
            Y_RIS = model.Codeword_activation([codebook_index, Channel_1])
            channel_h_rx = model.Tensor_multiplication_channel_h_rx(
                [H_RIS_to_rx, Y_RIS]
            )
            final_ris_signal = noise + channel_h_rx
            pred_final_signal = Decoder.predict(final_ris_signal)
            pred_output = np.argmax(pred_final_signal, axis=1)
            no_errors = pred_output != Testing_input
            no_errors = no_errors.astype(int).sum()
            ber[n] = no_errors / nn
            print("SNR:", EbNodB_range[n], "BER:", ber[n])
