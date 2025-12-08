import sys
import time

sys.path.append("./datasets")
import os
import numpy as np
import pickle as pkl
from datasets.data_preprocess import data_preprocess
from logistic_regression import LogisticRegression
from options import args_parser


current_work_dir = os.path.dirname(__file__)

if __name__ == "__main__":

    args = args_parser()

    (x_train, y_train), (x_test, y_test) = data_preprocess(args)

    print("learning rate: ", args.lr)
    print("Optimizer: ", args.optimizer.upper())

    Model = LogisticRegression(
        args=args,
        X_train=x_train,
        Y_train=y_train,
        X_test=x_test,
        prev_gradient=None,
        prev_update_direction=None,
        prev_weights=None
    )
    # start timer after cvx calculation done
    start_time = time.perf_counter()

    weight_diff_list = []
    obj_diff_list = []
    weight_diff, obj_diff = Model.diff_cal(Model.weights)
    print("\n------------ Initial ------------")
    print("weight error: {:.4e}".format(weight_diff))
    print("objective error: {:.4e}".format(obj_diff))

    Eigvals = np.linalg.eigvals(Model.pre_Hessian)
    print("\nmax eigenvalue of Hessian:{:.4f}".format(np.max(Eigvals)))
    print("min eigenvalue of Hessian:{:.4f}".format(np.min(Eigvals)))

    """
    update
    """

    for i in range(args.iteration):
        weight_diff, obj_diff = Model.update()

        print("\n------------ Iteration {} ------------".format(i + 1))
        print("weight error: {:.4e}".format(weight_diff))
        print("objective error: {:.4e}".format(obj_diff))
        (
            print("lambda value: {:.4e}".format(args.lambda_))
            if args.optimizer == "LM"
            else None
        )

        weight_diff_list.append(weight_diff)
        obj_diff_list.append(obj_diff)

        if weight_diff / np.sqrt(Model.dimension) <= 1e-5:
            break

    end_time = time.perf_counter()

    file_name = "./results/{}_{}.pkl".format("logreg", args.optimizer)
    file_name = os.path.join(current_work_dir, file_name)

    val = Model.getTest() > 0.5
    val2 = y_test > 0.5
    correct = 0

    for i in range(len(val2)):
        if val[i] == val2[i]:
            correct += 1

    percent_correct = correct / len(val2) * 100

    print(percent_correct, "Accuracy: %")
    print(f"\nTotal elapsed time: {(end_time - start_time):.2f} second")
    with open(file_name, "wb") as f:
        pkl.dump([weight_diff_list, obj_diff_list], f)
