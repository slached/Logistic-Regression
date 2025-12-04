import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", type=float, default=0.1, help="learning rate for each update step"
    )
    # GD(Gradient Descent) MN(Modified-Newton) LM(Levenberg-Marquardt) CG(Conjugate Gradient)
    parser.add_argument(
        "--optimizer", type=str, default="GD", help="using optimizer for update"
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
        default=1e-2,
        help="levenberg-marquardt's parameter for LM optimizer",
    )

    args = parser.parse_args()
    return args
