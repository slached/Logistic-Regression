import numpy as np
import cvxpy as cp

epsilon = 1e-5


class LogisticRegression:
    def __init__(self, args, X_train, Y_train, X_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

        self.num_samples = self.X_train.shape[0]
        self.dimension = self.X_train.shape[1]
        self.weights = np.zeros_like(X_train[0])

        self.args = args
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.gamma = args.gamma

        print("============= CVX solving =============")
        self.opt_weights, self.opt_obj = self.CVXsolve()
        print("============= CVX solved =============")

        self.pre_Hessian = self.Hessian(self.weights)

    def sigmoid(self, input):
        return 1.0 / (1.0 + np.exp(-input))

    def getTest(self):
        self.test = self.sigmoid(self.X_test @ self.weights)
        return self.test

    def objective(self, weights):
        """
        return the objective value of the problem
        note that the objective is averaged over all samples
        """
        sigWeights = self.sigmoid(self.X_train @ weights)
        matGrad = self.Y_train * np.log(sigWeights + epsilon) + (
            1.0 - self.Y_train
        ) * np.log(1 - sigWeights + epsilon)
        return (
            -np.sum(matGrad) / self.num_samples
            + 0.5 * self.gamma * np.linalg.norm(weights) ** 2
        )

    def gradient(self, weights):
        """
        return the gradient of objective function
        note that the gradient is averaged over all samples
        """
        sigWeights = self.sigmoid(self.X_train @ weights)
        matGrad = self.X_train.T @ (sigWeights - self.Y_train)
        return matGrad / self.num_samples + self.gamma * weights

    def Hessian(self, weights):
        """
        return the Hessian of objective function
        note that the Hessian is averaged over all samples
        """
        sigWeights = self.sigmoid(self.X_train @ weights)
        D_diag = np.diag(sigWeights * (1 - sigWeights))
        return (
            self.X_train.T @ D_diag @ self.X_train / self.num_samples
            + self.gamma * np.identity(self.dimension)
        )

    def update(self):
        """
        update model weights using GD  step
        """
        gradient = self.gradient(self.weights)

        if self.optimizer == "GD":
            update_direction = gradient
        elif self.optimizer == "MN":
            update_direction = (
                self.lr * np.linalg.inv(self.Hessian(self.weights)) @ gradient
            )
        elif self.optimizer == "LM":

            update_direction = (
                np.linalg.inv(
                    self.Hessian(self.weights)
                    + self.args.lambda_ * np.identity(self.dimension)
                )
                @ gradient
            )

            new_weight = self.weights - (self.lr * update_direction)
            # look if new weight has less error then divide the lambda by 2 else multiply by 2
            if self.objective(new_weight) < self.objective(self.weights):
                self.args.lambda_ /= 2
            else:
                self.args.lambda_ *= 2
        else:
            raise NotImplementedError

        self.weights -= self.lr * update_direction
        a, b = self.diff_cal(self.weights)
        return a, b

    def CVXsolve(self):
        """
        use CVXPY to solve optimal solution
        """
        x = cp.Variable(self.dimension)
        objective = cp.sum(
            cp.multiply(self.Y_train, self.X_train @ x) - cp.logistic(self.X_train @ x)
        )
        # ridge regression calculation
        l2 = 0.5 * self.gamma * cp.norm2(x) ** 2
        prob = cp.Problem(cp.Maximize(objective / self.num_samples - l2))
        prob.solve(solver=cp.ECOS_BB, verbose=False)  # False if not print it

        opt_weights = np.array(x.value)
        opt_obj = self.objective(opt_weights)

        return opt_weights, opt_obj

    def diff_cal(self, weights):
        """
        calculate the difference of input model weights with optimal in terms of:
        -   weights
        -   objective
        """
        weight_diff = np.linalg.norm(weights - self.opt_weights)
        obj_diff = abs(self.objective(weights) - self.opt_obj)
        return weight_diff, obj_diff
