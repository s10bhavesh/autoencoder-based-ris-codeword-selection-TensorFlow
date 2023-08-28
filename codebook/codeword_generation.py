from config import *
from model.channels_calculation import *
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt

Channel = CHANNELS()


class CODEBOOK_GENERATION:
    def __init__(self, H_tr_to_RIS) -> None:
        self.EbNo = np.power(10, (30 / 10))
        self.data = {}
        self.H_tr_to_RIS = H_tr_to_RIS

    def codebook_generation(self):
        """
        Generate all possible codewords, N^RIS_elements, for RIS.
        """
        x = [
            [list(i[x : x + RIS_rows]) for x in range(0, len(i), RIS_rows)]
            for i in product(["0", "180"], repeat=RIS_rows * RIS_cols)
        ]
        codebook = np.array(x, dtype="int64")

        Configurations = tf.dtypes.cast(
            tf.convert_to_tensor(
                codebook.reshape(2 ** (RIS_rows * RIS_cols), RIS_elements)
            ),
            tf.float64,
        )
        rad_to_deg = m.pi / 180
        code_radian = tf.math.scalar_mul(rad_to_deg, Configurations)
        phases = tf.complex(
            tf.math.cos(code_radian), tf.math.sin(code_radian), tf.complex128
        )
        return phases, codebook

    def calculate_SNR(self):
        """
        Calculate SNR

        """
        phases, codebook = self.codebook_generation()
        code_matrix = {}
        for angle in Azimuth_reflection:
            SNR = []
            H_ris_rx = Channel.Channel_values_generation(Azimuth_incidence, angle)
            distance = 1 / ((Dist_tx_ris * Dist_rx_ris) ** 2)
            H = tf.convert_to_tensor(np.multiply(self.H_tr_to_RIS, H_ris_rx))
            snr_value = np.multiply(
                self.EbNo, np.multiply(distance, (abs(tf.matmul(phases, H)) ** 2))
            )
            snr_db = np.round(10 * np.log10(snr_value), 12)

            for i in snr_db.flatten():
                SNR.append(i)
            code_matrix[angle] = SNR

        code_data = pd.DataFrame(code_matrix)
        code_data.to_csv("codebook/SNR_values.csv")
        return code_data

    def snr_sorting_and_filtering(self):
        """
            Sorting the SNR values in descending order and stored it in a \
            dictionary with its index values.
        """
        snr_values = self.calculate_SNR()
        sorted_code_SNR = {}
        for angle in Azimuth_reflection:
            data = {}
            ascend_data = snr_values.sort_values(angle, ascending=False)
            index = ascend_data.index.values.tolist()
            data["codeword"] = index
            data["SNR"] = ascend_data[angle].values
            sorted_code_SNR[angle] = data

        """ 
            Fix the SNR value as greater than tor equal to 10DB, and store\
            the sorted value with its index values.
        """
        sorted_snr_fix = {}
        for angle in Azimuth_reflection:
            data = {}
            ascend_data = snr_values.sort_values(angle, ascending=False)
            sorted_value = ascend_data[ascend_data[angle] >= 10]
            data["codeword"] = sorted_value.index.values.tolist()
            data["SNR"] = sorted_value[angle].values
            sorted_snr_fix[angle] = data

        code_snr_set = {}
        for angle in Azimuth_reflection:
            df = {}
            unq_snr = pd.DataFrame(
                {"SNR": sorted_snr_fix[angle]["SNR"]},
                index=sorted_snr_fix[angle]["codeword"],
            )
            unq = unq_snr[unq_snr["SNR"].duplicated(keep=False)]
            df = unq.groupby(list(unq)).apply(lambda x: set(sorted(x.index))).to_dict()
            code_snr_set[angle] = df

        codeword_set = pd.DataFrame(code_snr_set)
        codeword_set.index.name = "SNR"
        pd.DataFrame(codeword_set).to_csv("codebook/SNR_codeword_set.csv")
        return sorted_snr_fix

    def snr_preprocessing(self):
        """ """
        code_snr_set = {}
        sorted_snr = self.snr_sorting_and_filtering()
        for angle in Azimuth_reflection:
            df = {}
            unq_snr = pd.DataFrame(
                {"SNR": sorted_snr[angle]["SNR"]}, index=sorted_snr[angle]["codeword"]
            )
            unq = unq_snr[unq_snr["SNR"].duplicated(keep=False)]
            df = unq.groupby(list(unq)).apply(lambda x: set(sorted(x.index))).to_dict()
            code_snr_set[angle] = df

        codeword_set = pd.DataFrame(code_snr_set)
        codeword_set.index.name = "SNR"
        pd.DataFrame(codeword_set).to_csv("codebook/SNR_codeword_set.csv")
        # print(codeword_set)

        df_data = pd.read_csv("codebook/SNR_codeword_set.csv")
        max_snr = {}

        for i in Azimuth_reflection:
            data_index = df_data[df_data[str(i)].notnull()].index
            max = 0
            for j in data_index:
                if df_data["SNR"][j] >= max:
                    max = df_data["SNR"][j]
                    idx = j
            max_snr[i] = df_data[str(i)][idx]
        return max_snr

    def unique_codewords(self):
        """Make sets from the string loaded as csv file"""
        unique_set = dict()
        # print(self.snr_preprocessing(H_tr_to_RIS))
        for index, value in (self.snr_preprocessing()).items():
            if value not in unique_set:
                data = (value.replace("{", "").replace("}", "").replace(" ", "")).split(
                    ","
                )
                data_set = set()
                size = 0
                for x in data:
                    data_set.add(int(x))
                    size += 1
                unique_set[index] = data_set

        unique_codewords = list()
        for idx, val in unique_set.items():
            for x in val:
                if (x in unique_codewords) == False:
                    unique_codewords.append(x)
                    break
        unique_codewords
        return unique_codewords
