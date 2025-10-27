import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def plot_image(deposits: npt.NDArray, title: str = "title"):
        im = plt.imshow(
            deposits.reshape(18, 14), vmin=0, vmax=deposits.max(), cmap="Purples"
        )
        ax = plt.gca()
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(r"Calorimeter E$_T$ deposit (GeV)")
        plt.xticks(np.arange(14), labels=np.arange(4, 18))
        plt.yticks(
            np.arange(18),
            labels=np.arange(18)[::-1],
            rotation=90,
            va="center",
        )
        plt.xlabel(r"i$\eta$")
        plt.ylabel(r"i$\phi$")
        plt.title(title)
        plt.show()
        plt.close()
        

def plot_loss_history(training_loss: npt.NDArray, validation_loss: npt.NDArray):
    plt.plot(np.arange(1, len(training_loss) + 1), training_loss, label="Training")
    plt.plot(np.arange(1, len(validation_loss) + 1), validation_loss, label="Validation")
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    plt.close()
