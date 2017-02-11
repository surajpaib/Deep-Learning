from numpy import *
import matplotlib.pyplot as plt

def mean_squared_error(actual, predicted):
    return 0.5*((actual-predicted)**2)


def compute_cost(x, y, theta1, theta2, n):
    cost_function = 0
    for xi, yi in zip(x, y):
        pred_y = theta1 + (theta2 * xi)
        cost_function += mean_squared_error(yi, pred_y)

    cost_function /= n
    return cost_function


def gradient_descent_update(x, y, theta1, theta2, n, lr):
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
        for i in range(iterations):
            print "Training {0} / {1}".format(i, iterations)
            cost_function = compute_cost(self.x, self.y, theta1, theta2, self.length)
            [new_theta1, new_theta2] = gradient_descent_update(self.x, self.y, theta1, theta2, self.length, lr)
            theta1 = new_theta1
            theta2 = new_theta2
            print "Cost Function : {0} \t Updated theta1 : {1} \t theta2 : {2}".format(cost_function, theta1, theta2)
        self.theta1 = theta1
        self.theta2 = theta2

    def plot(self,time):
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
        theta1 = self.theta1
        theta2 = self.theta2
        y = theta1 + theta2 * x
        print "The predicted value for x = {0} is y = {1}".format(x, y)


def run():

    dataset = genfromtxt('challenge_dataset.txt', delimiter=',')
    ## Model Definition and functions from Linear Regression Class

    model = LinearRegression(dataset)
    model.plot(0.001)
    model.train(theta1=0, theta2=0, iterations=10000, lr=0.0001)
    model.plot(4)
    model.predict(0.5)



if __name__ == '__main__':
    run()
