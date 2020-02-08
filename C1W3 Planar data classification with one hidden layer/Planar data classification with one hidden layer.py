"""第三周编程作业
建立含有一个隐藏层的神经网络（与逻辑回归实现的模型有很大区别）

将会学会如何：
- 实现有一个隐藏层的二分类神经网络
- 使用非线性激活函数，eg. tanh
- 计算交叉熵损失
- 实现前向和后向传播
"""

# 导入包
import numpy as np                  # Python 进行科学计算的基本包
import matplotlib.pyplot as plt     # Python 中用于绘制图表的库
from testCases_v2 import *          # 提供一些测试样例来评估函数的正确性
import sklearn                      # 为数据挖掘和数据分析提供简单高效的工具
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# %matplotlib inline                # 使用 Jupyter Notebook 时使用


def layer_sizes(X, Y):
    """
    获取神经网络各层的神经元数
    :param X: 输入数据集，维度为（输入特征的数量，样本数量）
    :param Y: 标签，维度为（输出特征数量，样本数量）

    :return n_x: 输入层的神经元数
    :return n_h: 隐藏层的神经元数
    :return n_y: 输出层的神经元数
    """
    n_x = X.shape[0]    # 输入层的神经元数
    n_h = 4             # 隐藏层神经单元数固定为4
    n_y = Y.shape[0]    # 输出层的神经元数

    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    初始化模型参数
    :param n_x: 输入层（第 0 层）的神经元数
    :param n_h: 隐藏层（第 1 层）的神经元数
    :param n_y: 输出层（第 2 层）的神经元数

    :return parameters: 包含如下参数的字典
                    - W1: 第 1 层的权重，维度为(n_h, n_x)
                    - b1: 第 1 层的偏置，维度为(n_h, 1)
                    - W2: 第 2 层的权重，维度为(n_y, n_h)
                    - b2: 第 2 层的偏置，维度为(n_y, 1)
    """
    np.random.seed(2)   # 设定一个种子，使得我们的输出匹配，即便初始化是随机的

    # 初始化权重和偏置
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    # 使用断言确保数据维度是正确的
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2}

    return parameters


def forward_propagation(X, parameters):
    """
    前向传播
    :param X: 输入数据，维度为(n_x, m)
    :param parameters: 参数字典（初始化函数的输出）

    :return A2: 第二次激活的 sigmoid 输出
    :return cache: 缓存字典，包含如下
                    - Z1: Z1 = W1 X + b1
                    - A1: A1 = tanh(Z1)
                    - Z2: Z2 = W2 A1 + b2
                    - A2: A2 = sigmoid(Z2)
    """
    # 从参数字典中取出各层的权重和偏置
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # 前向传播
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    计算交叉熵成本
    :param A2: 第二次激活的 sigmoid 输出，维度为（1，样本数）
    :param Y: 标签，维度为（1，样本数）
    :param parameters: 包含 W1, b1, W2, b2 的字典

    :return cost: 交叉熵成本
    """
    m = Y.shape[1]  # 样本数

    # 计算交叉熵成本
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = -1 / m * np.sum(logprobs)

    cost = float(np.squeeze(cost))

    assert(isinstance(cost, float))

    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    后向传播
    :param parameters: 参数字典
    :param cache: 包含 Z1, A1, Z2, A2 的字典
    :param X: 输入数据，维度为（2，样本数）
    :param Y: 正确的标签数据，维度为（1，样本数）

    :return grads: 包含不同参数梯度的字典
    """
    m = X.shape[1]  # 样本数

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    # 后向传播，计算 dW1, db1, dW2, db2
    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads


def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    使用梯度下降法更新参数
    :param parameters: 参数字典
    :grads: 梯度字典
    :learning_rate: 学习率

    :return parameters: 更新后的参数字典
    """

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # 梯度下降法更新参数
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    神经网络模型
    :param X: 输入数据，维度为（2，样本数）
    :param Y: 标签数据，维度为（1，样本数）
    :param n_h: 隐藏层神经元数
    :param num_iterations: 梯度下降法更新参数的迭代次数
    :param print_cost: 若为 True ，则每 1000 次打印 cost

    :return parameters: 模型学习得到的参数字典，用于预测
    """
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]  # 输入层的神经元数
    n_y = layer_sizes(X, Y)[2]  # 输出层的神经元数

    # 初始化参数
    parameters = initialize_parameters(n_x, n_h, n_y)

    # 循环迭代（梯度下降法更新参数）
    for i in range(0, num_iterations):
        # 前向传播
        A2, cache = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y, parameters)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads)

        # 每 1000 次迭代打印 cost
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    """
    通过学习得到的参数，来预测 X 中样本的类别
    :param parameters: 参数字典
    :param X: 输入数据，维度为 (n_x, m)

    :return predictions: 模型的预测结果 (red: 0 / blue: 1)
    """

    # 利用前向传播计算概率
    A2, cache = forward_propagation(X, parameters)
    # 如果激活值（概率） > 0.5 ，则划分为类别 1 ， 否则为类别 0
    predictions = (A2 > 0.5)

    return predictions


def main():
    np.random.seed(1)   # 设置一个种子，使得结果是一致的

    # 导入花型的 2 类数据集
    # X: 输入数据集，维度为（输入特征的数量，样本数量）
    # Y: 标签数据集，维度为（输出特征的数量，样本数量）
    X, Y = load_planar_dataset()

    shape_X = X.shape
    shape_Y = Y.shape
    m = Y.shape[1]  # 训练样本数

    print("X 的维度：" + str(shape_X))
    print("Y 的维度：" + str(shape_Y))
    print("训练样本数：" + str(m))

    # 训练得到参数
    parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost = True)

    # 预测并绘制分类边界
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, np.squeeze(Y))
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()  # 否则不显示

    # 预测并打印准确率
    predictions = predict(parameters, X)
    print("准确率：%d" % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')


if __name__ == '__main__':
    main()