from config import *
from model.model import AUTOENCODER
import numpy as np
from keras.layers import Input, Dense, GaussianNoise,Dropout
from keras.models import Model

# AE = AUTOENCODER()
# Autoencoder = AE.autoencoder_model()

class test:
    def __init__(self) -> None:
        pass

    def testing():

        RIS_input_size = int(RIS_elements *4)
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
        ris = Model(ris_input,ris_6)
        ris.summary()

        # making decoder from full autoencoder
        decoded_input = Input(shape=(2,)) 
        decoder_1 = Autoencoder.layers[-2](decoded_input)
        decoder_2 = Autoencoder.layers[-1](decoder_1)
        Decoder = Model(decoded_input, decoder_2)
        Decoder.summary()

    def plot():
        # for plotting learned consteallation diagram
        scatter_plot = []
        for i in range(0, Num_symbols):
            print(i)
        temp = np.zeros(Num_symbols)
        temp[i] = 1
        print(np.expand_dims(temp, axis=0))
        scatter_plot.append(encoder.predict(np.expand_dims(temp,axis=0)))
        scatter_plot = np.array(scatter_plot)
        print (scatter_plot.shape)
        print(scatter_plot)

        # ploting constellation diagram
        import matplotlib.pyplot as plt
        # import matplotlib.cm as cm

        scatter_plot = scatter_plot.reshape(M,2,1)
        # colors = cm.rainbow(np.linspace(0, 1, len(scatter_plot)))
        plt.scatter(scatter_plot[:,0],scatter_plot[:,1])
        plt.axis((-2.5,2.5,-2.5,2.5))
        # plt.axis((-10.0,10.0,-10.0,10.0))
        plt.grid()
        plt.show()
        pass

    def frange(x, y, jump):
        while x < y:
            yield x
            x += jump

    def Testing(self):
        EbNodB_range = list(frange(-15,20,1))
        # EbNodB_range = list(frange(0,30,0.5))
        No_iter = 1
        BER_list = []
        # BER_list = []
        # for i in range(No_iter):
        print(f"{i}nd iteration\n")
        ber = [None]*len(EbNodB_range)
        for n in range(0,len(EbNodB_range)):
        # Ber = []
        EbNo = 10.0**(EbNodB_range[n]/10.0)
        noise_std = np.sqrt(1/(2*bits_channel*EbNo))
        noise_mean = 0
        no_errors = 0
        nn = test_sample
        noise = noise_std * np.random.randn(nn,n_channel)

        encoded_signal = encoder.predict(Test_data)                   # Encoder prediction
        Channel_1 = Channel_h_tr([h_tr, encoded_signal])              # Channel transmitter(tr)
        RIS_input = RIS_Input([h_recev, Channel_1])              # RIS
        ris_signal = ris.predict(RIS_input)                           # RIS Prediction
        codebook_index = K.argmax(ris_signal)                         # Argmax
        Y_RIS = Codeword_Activation([codebook_index,Channel_1])       # Codeword Activation
        channel_h_rx = Channel_h_rx([h_recev, Y_RIS])                    # Channel receiver (rx)
        final_ris_signal = noise + channel_h_rx                       # RIS output
        pred_final_signal = Decoder.predict(final_ris_signal)         # Decoder Prediction
        pred_output = np.argmax(pred_final_signal,axis=1)
        no_errors = (pred_output != Test_label)
        # no_errors =  no_errors.sum()
        no_errors =  no_errors.astype(int).sum()
        ber[n] = no_errors / nn
        print ('SNR:',EbNodB_range[n],'BER:',ber[n])
        # print()
        # Ber.append(ber[n])

        # BER_list.append(ber)

        # avg_sum = []
        # for i in range(len(BER_list[0])):
        #   sum = 0
        #   for j in range(len(EbNodB_range)):
        #     sum += float(BER_list[j][i])
        #   avg_sum.append(sum/No_iter)

        print(len(BER_list))

        np.set_printoptions(threshold=sys.maxsize)
        pred_output

        # ploting ber curve
        import matplotlib.pyplot as plt
        from scipy import interpolate
        plt.plot(EbNodB_range, ber, 'bo',label='Autoencoder(3,3)')
        plt.yscale('log')
        plt.xlabel('SNR Range')
        plt.ylabel('Block Error Rate')
        # plt.grid()
        plt.legend(loc='upper right',ncol = 1)

        count

        tf.math.bincount(count)

        codes, idx = tf.unique(count[0])
        codes

        # Plotting all the codebooks 
        sns.set()
        # f, ax = plt.subplots(figsize=(8, 8))
        colors = ["gray", "lightgray"]
        cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))
        # line = [219,235,435]
        for i in codes:
        # for i in values:
        phase = np.array(codebook[i])
        # print(phase)
        f, ax = plt.subplots(figsize=(8,8))
        ax = sns.heatmap(phase, cmap = cmap, square = True, linewidth=0.5, cbar_kws ={"shrink": .5})
        # sns.heatmap(phase, cmap="PiYG", linewidth=.5, vmin = -180, vmax = 180)
        # sns.heatmap(phase, linewidth=.5, vmin = -180, vmax = 180)

        # plt.show()
        colorbar = ax.collections[0].colorbar
        # colorbar.set_ticks([22.5,67.5,112.5,157.5,-22.5,-67.5,-112.5,-157.5])
        colorbar.set_ticks([0, 180])

        colorbar.set_ticklabels(['0','180'])
        plt.show()

        import pandas as pd
        df = pd.DataFrame(data = [EbNodB_range,ber]).T
        df.rename(columns={0:"SNR", 1:"BER"}, inplace=True)
        df.to_csv("SNR_90.csv")

        # import csv
        # with open('/content/drive/MyDrive/COMET Project(kh)/Autoencoder/Replicate/Codebook Design/Channel_codes.csv', 'a', encoding='UTF8') as f:
        #   writer = csv.writer(f)
        #   writer.writerow(row)