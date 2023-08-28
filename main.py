from config import *
from model.channels_calculation import CHANNELS
from model.model import AUTOENCODER
from codebook.codeword_generation import *
from train import *
from test import *
from keras.models import Model
import tensorflow as tf

Channel = CHANNELS()
Train = Train()
Test = Test()

# Generate the input symbol
Training_input = np.random.randint(Num_symbols, size=Training_sample_size)
Testing_input = np.random.randint(Num_symbols, size=Test_sample_size)

# Generate channel values, h_tr : transmitter to RIS and h_rx : RIS to receiver.
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

# Model Creation
model = AUTOENCODER(H_tr_to_RIS, H_RIS_to_rx, unique_codewords, unique_codebook)
Autoencoder = model.autoencoder_model()
Autoencoder.summary()

# Train the model
train = Train.train(Training_input, Autoencoder)
Train.plot(train)

# Test the model
Test_data = Test.Test(H_tr_to_RIS, H_RIS_to_rx, Testing_input, Autoencoder, model)
