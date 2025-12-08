import numpy as np
import cvxpy as cp
from collections import deque

epsilon = 1e-5


class LogisticRegression:
    def __init__(
        self,
        args,
        X_train,
        Y_train,
        X_test,
        prev_gradient,
        prev_update_direction,
        prev_weights,
    ):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

        self.prev_gradient = prev_gradient
        self.prev_update_direction = prev_update_direction
        self.prev_weights = prev_weights

        self.num_samples = self.X_train.shape[0]
        self.dimension = self.X_train.shape[1]
        self.weights = np.zeros_like(X_train[0])
        self.G = np.identity(self.dimension)

        self.args = args
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.gamma = args.gamma

        self.memory_size = self.args.memory_size
        # auto flush last item data structure
        # s differences of weights
        # y differences of gradient vectors
        self.s_history = deque(maxlen=self.memory_size)
        self.y_history = deque(maxlen=self.memory_size)
        self.rho_history = deque(maxlen=self.memory_size)

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
        # copy old weights
        old_weights = self.weights.copy()
        old_gradient = gradient.copy()

        if self.optimizer == "GD":
            update_direction = gradient
            self.weights -= self.lr * update_direction
        elif self.optimizer == "MN":
            update_direction = np.linalg.inv(self.Hessian(self.weights)) @ gradient
            self.weights -= self.lr * update_direction
        elif self.optimizer == "LM":
            update_direction = (
                np.linalg.inv(
                    self.Hessian(self.weights)
                    + (self.args.lambda_ * np.identity(self.dimension))
                )
                @ gradient
            )
            new_weight = self.weights - update_direction

            # look if new weight has less error then divide the lambda by 2 else multiply by 2
            if self.objective(new_weight) < self.objective(self.weights):
                self.args.lambda_ /= 2
            else:
                self.args.lambda_ *= 2

            # after updating lambda value we can set new weight value
            self.weights = new_weight
        elif self.optimizer == "CG":
            # this means first iteration
            if self.prev_gradient is None:
                update_direction = -gradient
            else:
                alpha = (
                    np.linalg.norm(gradient) ** 2
                    / np.linalg.norm(self.prev_gradient) ** 2
                )
                update_direction = -gradient + alpha * self.prev_update_direction

            self.weights += self.lr * update_direction
        elif self.optimizer == "LBFGS":
            update_direction = self.two_loop_recursion(gradient=gradient)

            self.weights += self.lr * update_direction

            new_grad = self.gradient(self.weights)

            # for memory save
            s = self.weights - old_weights
            y = new_grad - old_gradient

            sy_dot = np.dot(s, y)
            # for safety steps
            if sy_dot > 1e-10:
                rho = 1.0 / sy_dot

                self.s_history.append(s)
                self.y_history.append(y)
                self.rho_history.append(rho)

        else:
            raise NotImplementedError

        a, b = self.diff_cal(self.weights)

        # save prev states for next iteration
        self.prev_gradient = old_gradient
        self.prev_update_direction = update_direction
        self.prev_weights = old_weights

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

    def two_loop_recursion(self, gradient):
        # q is old gradient
        q = gradient.copy()
        alpha_list = [0] * len(self.s_history)

        # Backward from newest to oldest
        for i in range(len(self.s_history) - 1, -1, -1):
            s = self.s_history[i]
            y = self.y_history[i]
            rho = self.rho_history[i]
            # check similarities of gradient vector and weighs differences
            alpha = rho * np.dot(s, q)

            alpha_list[i] = alpha
            # subtract little piece of change of gradient from current gradient vector
            q = q - alpha * y

        # Scaling
        # Initial Hessian Approximation
        # H0 = (s_last * y_last) / (y_last * y_last) * I
        if len(self.s_history) > 0:
            s_last = self.s_history[-1]
            y_last = self.y_history[-1]
            gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
            z = q * gamma
        else:
            # if memory is empty
            z = q

        # Forwarding
        # From oldest to newest
        for i in range(len(self.s_history)):
            s = self.s_history[i]
            y = self.y_history[i]
            rho = self.rho_history[i]
            alpha = alpha_list[i]
            # z is scaled and peeled gradient vector
            # beta is similarity of this scaled gradient and difference of gradients
            beta = rho * np.dot(y, z)
            # finally we can add to our scaled gradient weight diff * alpha - beta
            z = z + s * (alpha - beta)
        # this is approximately -G and we barely use memory(in this case 3 deque list of element of 10)
        return -z
