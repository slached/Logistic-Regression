import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
from options import args_parser
from matplotlib import rcParams

rcParams.update({"font.size": 18, "text.usetex": True})


def plot_logreg(optimizer):
    """
    logistic regression: weights
    """

    try:
        logreg_weights, logreg_objective = pkl.load(
            open(f"./results/logreg_{optimizer}.pkl", "rb")
        )
    except FileNotFoundError:
        raise FileNotFoundError

    logreg_dimension = 785
    plt.figure()
    plt.plot(
        range(len(logreg_weights)),
        np.array(logreg_weights) / np.sqrt(logreg_dimension),
        label=optimizer,
    )

    plt.legend()

    plt.xlabel("Iterations")
    plt.ylabel(r"$\frac{1}{\sqrt{d}}\|x^{(k)}-x^{\star}\|_2$")
    plt.title("Logistic Regression")

    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"./results/logreg_weights_{optimizer}.png", dpi=1200)
    plt.show()
    plt.pause(5)

    """
    logistic regression: objective
    """

    plt.figure()
    plt.plot(range(len(logreg_objective)), logreg_objective, label=optimizer)

    plt.legend()

    plt.xlabel("Iterations")
    plt.ylabel(r"$f(x^{(k)}) - p^{\star}$")
    plt.title("Logistic Regression")

    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"./results/logreg_objective_{optimizer}.png", dpi=1200)
    plt.show()


if __name__ == "__main__":
    args = args_parser()

    plot_logreg(args.optimizer)
