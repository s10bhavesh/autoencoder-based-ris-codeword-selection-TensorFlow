from codeword_generation import *


def density_plot_snr():
    """
    Density plot for the SNR.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(
        code_matrix.keys(), code_matrix.values(), marker=".", ms=10, linestyle="none"
    )
    plt.xlabel("Angles")
    plt.xticks(Azimuth_reflection, Azimuth_reflection)
    plt.ylabel("SNR Values")
    plt.title("Density plot for SNR values at each angles")
    plt.show()
