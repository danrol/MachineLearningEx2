import numpy as np
import matplotlib.pyplot as plt
import random


class MultiClassLR(object):
    def __init__(self, n_in, n_out):
        """
        ctor
        :param n_in: input dim
        :param n_out: output dim
        """
        self.W = np.random.randn(n_in, n_out)
        self.b = np.random.randn(n_out)

    def softmax(self, x):
        """
        softmax
        :param x: input vec
        :return: softmax vec
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def classifier_output(self, x):
        """
        classifier output
        :param x: input vec
        :return: p(y|x)
        """
        return self.softmax(np.dot(x, self.W) + self.b)

    def predict(self, x):
        """
        predicts label
        :param x: input vec
        :return: y hat
        """
        return np.argmax(self.classifier_output(x))

    def train(self, x, y, lr=0.1, L2_reg=0.100):
        """
        trains classifier
        :param x: input
        :param y: tabel
        :param lr: learning rate
        :param L2_reg: L2 regression
        :return: loss
        """
        self.x = x
        self.y = int(y)

        y_hat = self.predict(self.x)

        grad = -self.classifier_output(self.x)[0]
        grad[self.y-1] += 1

        self.W += lr * grad * self.x + lr * L2_reg * self.W
        self.b += lr * grad

        loss = -np.log(self.classifier_output(self.x)).sum()
        return loss


def gen_train_data(example_num):
    """
    generates data
    :param example_num: number of examples
    :return: training data
    """
    args1 = (np.random.normal(2 * 1, 1, example_num), np.random.normal(2 * 2, 1, example_num),
             np.random.normal(2 * 3, 1, example_num))
    x = np.concatenate(args1)
    args2 = (np.ones(example_num), 2 * np.ones(example_num), 3 * np.ones(example_num))
    y = np.concatenate(args2)
    zipped = zip(x, y)
    # for a,b in zipped:
    #     print(f'a = {a}, b = {b}')
    train_examples = zipped
    #random.shuffle(train_examples)
    np.random.shuffle(list(train_examples))
    return train_examples


def density_func(x, mu):
    """
    density function
    :param x: input
    :param mu: expectation
    :return: y based on density function
    """
    return (float(1) / np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / 2)


if __name__ == "__main__":
    # training data
    learning_rate = 0.01
    n_epochs = 100
    train_set = gen_train_data(100)

    #for a,b in train_set:
    #    print(f'a = {a}, b = {b}')
    # construct MultiClassLR
    classifier = MultiClassLR(n_in=1, n_out=3)

    # train
    loss = prev_loss = 10000
    for epoch in range(n_epochs):
        loss = 0
        examples = 0
        for (x, y) in train_set:
            examples += 1
            loss += classifier.train(x, y, lr=learning_rate)
            # print(f'examples={examples}')
        # if loss > prev_loss:
        #     break
        prev_loss = loss
        print(f'examples={examples}')
        print ('epoch:', epoch, ' loss:', loss / examples)
        learning_rate *= 0.9

    # test
    m = 10
    test_set = gen_train_data(m)
    sum = 0
    for (x, y) in test_set:
        if y != classifier.predict(x) + 1:
            sum += 1
    print (float(sum)/(m*3))

    # plot true
    ys_true = []
    ys_est = []
    xs = np.linspace(0, 10, 100).tolist()
    for x in xs:
        ys_true.append(density_func(x, 2 * 1) / (density_func(x, 2 * 1) + density_func(x, 2 * 2) + density_func(x, 2 * 3)))
        ys_est.append(classifier.classifier_output(x)[0][0])
    plt.figure()
    plt.suptitle('posterior probability')
    plt.xlabel('x')
    plt.ylabel('p(y=1|x)')
    plt.plot(xs, ys_true, 'r.', xs, ys_est, 'b.')
    plt.legend(['true distribution', 'logistic regression'])
    plt.savefig('dist.png')