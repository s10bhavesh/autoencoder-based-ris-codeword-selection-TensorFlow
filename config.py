import numpy as np

Num_symbols = 4  # no of possible symbols (e,g. 0,1,2,3), M
Num_symbol_bits = int(
    np.log2(Num_symbols)
)  # no of bits to represent all the symbols, k
Num_of_channel = 2  # no of channels, n_channels
Num_bits_per_channel = Num_symbol_bits / Num_of_channel  # no of bits per channel
Frequency = 5.8 * (10**9)  # Frequency in GHz
Lights_speed = 3 * (10**8)  # speed of light, c
Wavelength = Lights_speed / Frequency  # wavelength, Lambda
Insertion_loss = np.power(10, (3 / 10))  # insertion loss as 3db
RIS_element_gap = Wavelength / 2  # distance between two RIS elements, df
EbNo_train = np.power(10, (8 / 10))  #  coverted 10 db of EbNo
# EbNo_train = 5.01187 #  coverted 7 db of EbNo
Num_codebooks = 3  # number of codebook, No_codebook
Angle_count = 0
Cnt_ebno = 0

# No rows and columns of the RIS plate
RIS_rows = 3
RIS_cols = 3
RIS_elements = RIS_rows * RIS_cols
RIS_elements

Batch_size = 32

# Angles of elevation and azimuth for each elements
Eleveation_incidence = -27.5  # Eleveation angle of incidence, theta_i
Eleveation_reflection = (
    -27.5
)  # Eleveation angle of reflection (two degrees were best : 84 & 90 ), theta_d
Azimuth_incidence = 90  # Azimuth angle of incidence, phi_i
angles = []

for i in np.arange(90, 181, 5):  # start, end and gap
    angles.append(i)  # theta d varies for every 6 degrees in the
Azimuth_reflection = np.array(angles)

# Total Number of training symbols
Training_sample_size = 10000  # No of Training sample size
Test_sample_size = 10000

Dist_tx_ris = 5  # in Meter's
Dist_rx_ris = 10
