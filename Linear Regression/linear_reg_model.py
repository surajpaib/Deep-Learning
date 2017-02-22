from numpy import genfromtxt, zeros
import matplotlib.pyplot as plt
from metrics import mean_squared_error


def calculate_tss(y, n):
    """
    Calculates the Total Squared Sum
    :param y:
    :param n:
    :return:
    """
    y_avg = 0
    y_tss = 0
    for yi in y:
        y_avg += yi
    y_avg /= n

    for yi in y:
        y_tss += mean_squared_error(yi, y_avg)
    print y_tss
    return y_tss


def compute_cost(x, y, theta1, theta2, n):
    """
    Computes the cost function
    :param x:
    :param y:
    :param theta1:
    :param theta2:
    :param n:
    :return:
    """
    cost_function = 0
    for xi, yi in zip(x, y):
        pred_y = theta1 + (theta2 * xi)
        cost_function += mean_squared_error(yi, pred_y)

    cost_function /= n
    return cost_function


def gradient_descent_update(x, y, theta1, theta2, n, lr):
    """
    Does Gradient Descent Update, Refer to parameters from train function
    :param x:
    :param y:
    :param theta1:
    :param theta2:
    :param n:
    :param lr:
    :return:
    """
    theta1_grad = 0
    theta2_grad = 0
    for xi, yi in zip(x, y):
        theta1_grad += - (yi - (theta1 + theta2 * xi))
        theta2_grad += - (yi - (theta1 + theta2 * xi)) * xi

    theta1_grad /= n
    theta2_grad /= n
    theta1 -= lr * theta1_grad
    theta2 -= lr * theta2_grad
    return theta1, theta2


class LinearRegression:
    """
    Linear Regression Class
    """
    def __init__(self, dataset):
        self.x = dataset[:, 0]
        self.y = dataset[:, 1]
        self.theta1 = 0
        self.theta2 = 0
        self.iterations = 0
        self.lr = 0
        self.cost_function = 0.0
        self.theta1_grad = 0
        self.theta2_grad = 0
        self.length = len(self.y)

    def train(self, theta1, theta2, iterations, lr):
        """

        :param theta1: Weight parameter
        :param theta2: Weight parameter
        :param iterations: Number of Iterations
        :param lr: Learning Rate
        :return: Trains the model
        """
        for i in range(iterations):
            print "Training {0} / {1}".format(i, iterations)
            cost_function = compute_cost(self.x, self.y, theta1, theta2, self.length)
            [new_theta1, new_theta2] = gradient_descent_update(
                self.x, self.y, theta1, theta2, self.length, lr)
            theta1 = new_theta1
            theta2 = new_theta2
            print "Cost Function : {0} \t Updated theta1 :" \
                  " {1} \t theta2 : {2}".format(cost_function, theta1, theta2)
        self.theta1 = theta1
        self.theta2 = theta2

    def plot(self, time):
        """
        Plots x vs y predicted
        :param time:
        :return: plots
        """
        plt.ion()

        x = self.x
        y = self.y
        theta1 = self.theta1
        theta2 = self.theta2
        y_pred = theta1 + theta2 * x
        plt.scatter(x, y)
        plt.pause(0.0001)
        plt.plot(x, y_pred)
        plt.pause(time)

    def predict(self, x):
        """

        :param x:
        :return: predicted value of y
        """
        theta1 = self.theta1
        theta2 = self.theta2
        y = theta1 + theta2 * x
        print "The predicted value for x = {0} is y = {1}".format(x, y)

    def score(self):
        """

        :return: R squared error
        """
        x = self.x
        y = self.y
        theta1 = self.theta1
        theta2 = self.theta2
        sse = compute_cost(x, y, theta1, theta2, 1)
        print sse
        tss = calculate_tss(y, self.length)
        r_squared = 1 - (sse / tss)
        print (" The R-squared error is {0}".format(r_squared))

    def plot_residual(self):
        """

        :return: Plot of residuals
        """
        x = self.x
        y = self.y
        theta1 = self.theta1
        theta2 = self.theta2
        y_pred = theta1 + theta2 * x
        residual = y - y_pred
        plt.ion()
        plt.clf()
        plt.pause(0.01)
        plt.plot(x, zeros(97), '-+')
        plt.scatter(x, residual)
        plt.pause(5)


def run():
    """
    Runs the program
    Gets data from "challenge_dataset.txt"
    Calls the Linear Regression Class.
    """
    dataset = genfromtxt('challenge_dataset.txt', delimiter=',')
    # Model Definition and functions from Linear Regression Class

    model = LinearRegression(dataset)
    model.plot(0.001)
    model.train(theta1=0, theta2=0, iterations=10000, lr=0.0001)
    model.plot(4)
    model.score()
    model.plot_residual()


if __name__ == '__main__':
    run()
