"""
建立深度神经网络（你想要多少层就可以键多少层），用于图片分类
"""

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from testCases_v4a import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
from dnn_app_utils_v3 import *

np.random.seed(1)


def initialize_parameters_deep(layer_dims):
    """
    初始化参数
    :param layer_dims: 包含深度神经网络各层的 list
    
    :return parameters: 字典，包含神经网络各层的权重 'Wl' 和 偏置 'bl'
    """
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims) # 神经网络的层数

    # 初始化参数
    for l in range(1, L):
        # parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    神经网络一层的前向传播的线性部分 Z = WA + b
    :param A: 前一层的激活值
    :param W: 当前层的权重
    :param b: 当前层的偏置

    :return Z: 当前层激活前的参数，即激活函数的输入值
    :return cache: 线性缓存 (linear_cache)，包含 'A', 'W', 'b' 的元组，存储用于高效地计算后向传播
    """
    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)   # 前一层的激活值，当前层的权重，当前层的偏置

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    神经网络一层 (LINEAR->ACTIVATION) 的前向传播
    :param A_prev: 前一层的激活值
    :param W: 当前层的权重
    :param b: 当前层的偏置
    :param activation: 当前层的激活函数，string: "sigmoid" or "relu"
    
    :return A: 当前层激活函数的输出值
    :return cache: 元组，包含 "linear_cache" and "activation_cache (Z)", 存储用来高效的计算后向传播
    """
    # 激活函数为 sigmoid()
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)    # activation_cache 存储的为 Z
    # 激活函数为 relu()
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)       # activation_cache 存储的为 Z

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    深度神经网络 [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID 的前向传播计算
    :param X: 输入数据
    :param parameters: 初始化后的参数，即initialize_parameters_deep(layer_dims)的输出，包含各层的权重 Wl 和偏置 bl

    :return AL: 最后一层/输出层的激活值
    :return caches: list，包含 L 层 linear_activation_forward() 的返回值 cache:
                        - linear_relu_forward() 的 cache，L-1 个，索引为 0 - L-2
                        - linear_sigmoid_forward() 的cache，1 个，索引为 L-1
    """
    caches = []                 # 缓存
    A = X                       # A0 = X
    L = len(parameters) // 2    # 神经网络的层数

    # 实现 [LINEAR->RELU]*(L-1)
    for l in range(1, L):
        A_prev = A              # 前一层的激活值
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)

    # 实现 LINEAR->SIGMOID
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)

    assert(AL.shape == (1,X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):
    """
    计算 cost J
    :param AL: 与标签预测对应的概率向量，即最后一层的激活值输出
    :param Y: 正确的标签向量

    :return cost: 交叉熵代价
    """
    m = Y.shape[1]

    # 计算交叉熵损失值
    cost = -1 / m * np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), (1 - Y)))

    cost = np.squeeze(cost) # 保证 cost 的维度
    assert(cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    """
    神经网络一层 (layer l) 的后向传播的线性部分
    :param dZ: 当前层线性部分的输出的梯度
    :param cache: linear_cache，元组 (A_prev, W, b), 存储与当前层的前行传播
    
    :rerurn dA_prev: 前一层的激活值的梯度
    :return dW: 当前层的权重的梯度
    :return db: 当前层的偏置的梯度
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]     # 样本数

    # 计算梯度
    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    神经网络一层，即 LINEAR->ACTIVATION layer 的后向传播
    :param dA: 当前层激活值的梯度
    :param cache: 元组 (linear_cache, activation_cache)
    :param activation: 当前层使用的激活函数，string: "sigmoid" or "relu"

    :return dA_prev: 前一层激活值的梯度
    :return dW: 当前层的权重的梯度
    :return db: 当前层的偏置的梯度
    """
    linear_cache, activation_cache = cache  # 线性部分的缓存，激活部分的cache

    # 当前层的激活函数为 relu()
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    # 当前层的激活函数为 sigmoid()
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    深度神经网络 [LINEAR->RELU]*(L-1) -> LINEAR->SIGMOID 的后向传播
    :param AL: 概率向量，即神经网络前向传播的输出/最后一层的激活值
    :param Y: 正确的标签向量(0:non-cat / 1:cat)
    :param caches: 缓存list，包含：
                    - linear_relu_forward() 的 cache，L-1 个，索引为 0 - L-2
                    - linear_sigmoid_forward() 的cache，1 个，索引为 L-1
    
    :return grads: 梯度字典
                    - grads["dA" + str(l)] = ...
                    - grads["dW" + str(l)] = ...
                    - grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)         # 神经网络的层数
    m = AL.shape[1]         # 样本数
    Y = Y.reshape(AL.shape) # 使得 Y 的维度同 AL 一样

    # 初始化后向传播，计算dAL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # 计算第 L 层 (SIGMOID->LINEAR) 的梯度（即最后一层的梯度）
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")

    # 计算 l = L-1 到 1 层 (RELU->LINEAR) 的梯度，从 L-2 循环到 0
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], current_cache, 'relu')
        grads['dA' + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    梯度下降法更新参数，一轮
    :param parameters: 参数字典
    :param grads: 梯度字典
    :param learning_rate: 学习率

    :return parameters: 更新后的参数字典（包括权重 Wl 和偏置 bl）
    """
    L = len(parameters) // 2    # 神经网络的层数

    # 更新各层的参数
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    L 层的神经网络模型：[LINEAR->RELU]*(L-1) -> LINEAR->SIGMOID
    :param X: 输入数据
    :param Y: 正确的标签向量
    :param layers_dims: 各层的神经元数
    :param learning_rate: 学习率
    :param num_iterations: 优化迭代次数
    :param print_cost: 若为 True ，则每 100 次迭代打印 cost
    """
    np.random.seed(1)
    costs = []  # 记录 cost

    # 初始化参数
    parameters = initialize_parameters_deep(layers_dims)
    
    # 梯度下降法迭代更新参数
    for i in range(0, num_iterations):
        # 前向传播 [LINEAR->RELU]*(L-1) -> LINEAR->SIGMOID
        AL, caches = L_model_forward(X, parameters)

        # 计算代价
        cost = compute_cost(AL, Y)
    
        # 后向传播
        grads = L_model_backward(AL, Y, caches)
 
        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # 每迭代 100 次打印 cost
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # 将 cost 变化绘制成图
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


def predict(X, y, parameters):
    """
    预测
    :param X: 要预测（标记）的数据集
    :param y: 正确的标签向量
    :param parameters: 神经网络模型训练得到的参数
    
    :return p: 给定数据集的预测结果
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p


def main():
    # 导入数据集
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    # reshape 训练和测试样本 --> (R+G+B, m)
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # 标准化数据，使特征值在 0-1 之间
    train_x = train_x_flatten / 255
    test_x = test_x_flatten / 255

    # print ("train_x's shape: " + str(train_x.shape))
    # print ("test_x's shape: " + str(test_x.shape))

    # 神经网络各层的神经元数，这里为 4 层模型
    layers_dims = [12288, 20, 7, 5, 1]

    # 开始训练参数，训练过程中会打印 cost 参数变化即变化图
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

    # 预测
    pred_train = predict(train_x, train_y, parameters)  # 对训练样本进行预测
    pred_test = predict(test_x, test_y, parameters)     # 对测试样本进行预测


if __name__ == '__main__':
    main()