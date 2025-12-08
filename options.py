import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", type=float, default=0.1, help="learning rate for each update step"
    )
    # gd(Gradient Descent) mn(Modified-Newton)
    # lm(Levenberg-Marquardt) cg(Conjugate Gradient)
    # lbfgs(Limited-memory BFGS)
    parser.add_argument(
        "--optimizer", type=str, default="lbfgs".upper(), help="using optimizer for update"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=500,
        help="maximum update iterations if not exit automatically",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="penalty term for logistic regression",
    )
    parser.add_argument(
        "--lambda_",
        type=float,
        default=1000,
        help="levenberg-marquardt's parameter for LM optimizer",
    )
    parser.add_argument(
        "--memory_size",
        type=float,
        default=10,
        help="memory size for lbfgs",
    )

    args = parser.parse_args()
    return args
