from config import *
from model.channels_calculation import CHANNELS
from model.model import AUTOENCODER
from codebook.codeword_generation import *
from train import *

# from test import *
from keras.models import Model
import tensorflow as tf


Channel = CHANNELS()
Train = Train()


#
#     Generate the input symbols.
#     M : Number of the possible symbols,
#     N : Number of samples.
#
Training_input = np.random.randint(Num_symbols, size=Training_sample_size)
# Input_data = tf.one_hot(input, Num_symbols, dtype=tf.int64)

#
#     Generate the channel values -
#     h_tr : Transmitter to RIS channel,
#     h_rx : RIS to receiver channel.
#
H_tr_to_RIS = Channel.Channel_values_generation(Eleveation_incidence, Azimuth_incidence)
H_RIS_to_rx = Channel.Channel_values_generation(Eleveation_reflection, 90)

# Codebook generation
Codebook = CODEBOOK_GENERATION(H_tr_to_RIS)
phases, All_codebook = Codebook.codebook_generation()
unique_codewords = Codebook.unique_codewords()

unique_codebook = list()
for i in unique_codewords:
    unique_codebook.append(All_codebook[i])
unique_codebook = np.array(unique_codebook)

#
#     Create a model and train it.
#
model = AUTOENCODER(H_tr_to_RIS, H_RIS_to_rx, unique_codewords, unique_codebook)
autoencoder = model.autoencoder_model()
autoencoder.summary()

train = Train.train(Training_input, autoencoder)
Train.plot(train)

test = np.random.randint(
    Num_symbols, size=Test_sample_size
)  # Generate test_sample no of random numbers
# Test_data = tf.one_hot(test, Num_symbols, dtype=tf.int64)
# Test_label = test

# Autoencoder = Model(input_signal, Decoded)
# Autoencoder.compile(optimizer=Adam(learning_rate = 0.01), loss='categorical_crossentropy',metrics=['accuracy'])
# Autoencoder.summary()
