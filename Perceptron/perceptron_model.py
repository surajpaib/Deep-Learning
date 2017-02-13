from numpy import dot, random, exp, array, transpose


class PerceptronNetwork:
    """

    """
    def __init__(self):
        """

        """
        random.seed(1)
        self.weights = 2 * random.random((3, 1)) - 1

    def _sigmoid(self, x):
        """

        :param x:
        :return:
        """
        return 1/(1 + exp(-x))

    def _sigmoid_derivative(self, x):
        """

        :param x:
        :return:
        """
        return x * (1 - x)

    def predict(self, x):
        """

        :param x:
        :return:
        """
        return self._sigmoid(dot(x, self.weights))

    def train(self, x, y, iterations):
        """

        :param x:
        :param y:
        :param iterations:
        :return:
        """
        for i in range(iterations):
            print " Perceptron Training.... {0} / {1}".format(i, iterations)
            y_pred = self.predict(x)
            error = y - y_pred
            weight_update = dot(transpose(x), error * self._sigmoid_derivative(y_pred))
            self.weights += weight_update
            print " Error is {0}, Updated weight: {1}".format(error, self.weights)

    def test(self, x):
        """

        :param x:
        :return:
        """
        y_test = self.predict(x)
        print "Predicted output is {0}".format(y_test)


def run():
    """

    :return:
    """
    neural_net = PerceptronNetwork()

    training_inputs = array([[1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0]])
    training_outputs = array([[1, 0, 0, 1]]).T
    test_input = array([1, 1, 1])
    neural_net.train(training_inputs, training_outputs, iterations=1000)
    neural_net.test(test_input)


if __name__ == '__main__':
    run()
