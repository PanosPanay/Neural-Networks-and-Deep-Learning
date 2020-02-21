"""
神经网络的参数初始化
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

# %matplotlib inline  #如果你使用的是Jupyter Notebook，请取消注释
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    """
    实现一个 3 层的神经网络: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    :param X: 输入数据, of shape (2, number of examples)
    :param Y: 正确的标签向量 (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    :param learning_rate: 梯度下降的学习率 
    :param num_iterations: 梯度下降的迭代次数
    :param print_cost: if True, print the cost every 1000 iterations
    :param initialization: string, 决定要使用的初始化方法 ("zeros","random" or "he")
    
    :return parameters: 神经网络学习到的参数
    """
        
    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]
    
    # 初始化参数字典
    if initialization == "zeros":       # 0 初始化
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":    # 随机初始化成较大的权重
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":        # 随机初始化成较小的权重，规模依据 paper by He et al., 2015.
        parameters = initialize_parameters_he(layers_dims)

    # 梯度下降法迭代
    for i in range(0, num_iterations):

        # 前向传播: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)
        
        # 计算损失值
        cost = compute_loss(a3, Y)

        # 后向传播
        grads = backward_propagation(X, Y, cache)
        
        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
            
    # 绘制成本曲线（cost 值随着迭代的变化）
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


def initialize_parameters_zeros(layers_dims):
    """
    0 初始化，将参数全部初始化为0
    :param layer_dims: python array (list), 包含各层的神经元数
    
    :return parameters: python dictionary, 包含参数： "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    parameters = {}
    L = len(layers_dims)            # 神经网络的层数
    
    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###
    return parameters


def initialize_parameters_random(layers_dims):
    """
    随机初始化。将权重初始化为较大的随机值（*10），将偏置初始化为 0
    :param layer_dims: python array (list), 包含各层的神经元数
    
    :return parameters: python dictionary, 包含参数： "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    L = len(layers_dims)            # 神经网络的层数
    
    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###

    return parameters


# 最优方案
def initialize_parameters_he(layers_dims):  
    """
    He 初始化。将权重初始化为较小的值（*sqrt(2./layers_dims[l-1])），将偏置初始化为 0
    :param layer_dims: python array (list), 包含各层的神经元数
    
    :return parameters: python dictionary, 包含参数： "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)             # 神经网络的层数
     
    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###
        
    return parameters


def main():
    # 读取并绘制数据: blue/red dots in circles
    train_X, train_Y, test_X, test_Y = load_dataset()

    # 选择一个初始化方法
    parameters = initialize_parameters_he([2, 4, 1])

    # 打印随机初始化后各参数的值
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    # 通过神经网络模型进行训练，得到参数
    parameters = model(train_X, train_Y, initialization = "he")

    # 用训练得到的参数进行预测，并分别输出训练集和测试集的预测准确率
    print ("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    # 打印预测结果
    print (predictions_train)
    print (predictions_test)

    # 绘制红蓝点分离情况图
    plt.title("Model with He initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, np.squeeze(train_Y))    # 注意最后一个参数不能直接用train_Y，要加上np.squeeze() 


if __name__ == '__main__':
    main()