"""
正则化，避免过拟合（高方差）

未解决 bug：FileNotFoundError: [Errno 2] No such file or directory: 'data.mat'
"""

# import packages
import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

# %matplotlib inline  #如果你使用的是Jupyter Notebook，请取消注释
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
    """
    实现一个三层神经网络模型: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    可以：-- 不使用正则化
          -- 使用 L2 正则化
          -- 使用 dropout 正则化

    :param X: 输入数据, of shape (input size, number of examples)
    :param Y: 正确的标签向量 (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    :param learning_rate: 优化的学习率
    :param num_iterations: 优化循环的迭代次数
    :param print_cost: If True, print the cost every 10000 iterations
    :param lambd: 正则化的超参数 λ
    :param keep_prob: drop-out 正则化中随机保留节点的概率
    
    Returns:
    parameters -- 神经网络模型学习得到的参数，可以用来预测
    """
        
    grads = {}
    costs = []                            # to keep track of the cost
    m = X.shape[1]                        # number of examples
    layers_dims = [X.shape[0], 20, 3, 1]
    
    # 初始化参数字典
    parameters = initialize_parameters(layers_dims)

    # 学习循环（梯度下降）
    for i in range(0, num_iterations):

        # 前向传播: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        # 不使用 dropout 正则化
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        # 使用 dropout 正则化
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        
        # 计算 cost
        # 不使用 L2 正则化
        if lambd == 0:
            cost = compute_cost(a3, Y)
        # 使用 L2 正则化
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
            
        # 后向传播
        assert(lambd==0 or keep_prob==1)    # 可以同时使用 L2 正则化和 dropout, 但是本程序只使用其中一个或不使用正则化
        # 不使用正则化                                    
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        # 使用 L2 正则化
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        # 使用 dropout 正则化
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        
        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
    
    # 绘制 cost 变化曲线图
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# 使用 L2 正则化
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    计算使用 L2 正则化的损失函数值
    
    :param A3: 正向传播的输出结果, of shape (output size, number of examples)
    :param Y: 正确的标签向量, of shape (output size, number of examples)
    :param parameters: python dictionary，包含模型的参数
    
    :return cost: 使用 L2 正则化的损失函数值
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    # 计算交叉熵部分的损失值
    cross_entropy_cost = compute_cost(A3, Y) 
    
    # 计算 L2 正则化部分的损失值
    ### START CODE HERE ### (approx. 1 line)
    L2_regularization_cost = lambd / (m * 2) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    ### END CODER HERE ###
    
    # 最终的损失值
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost


def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    使用 L2 正则化的后向传播
    
    :param X: 输入数据集, of shape (input size, number of examples)
    :param Y: 正确的标签向量, of shape (output size, number of examples)
    :param cache: 正向传播的 cache 输出
    :param lambd: L2 正则化的超参数 λ
    
    :return gradients: 包含了每个参数W/b、激活值A、预激活值Z的梯度的字典
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    
    # 计算权重 W 的梯度时，要加上正则项的梯度值 λ / m * W
    ### START CODE HERE ### (approx. 1 line)
    dW3 = 1./m * np.dot(dZ3, A2.T) + lambd / m * W3
    ### END CODE HERE ###
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW2 = 1./m * np.dot(dZ2, A1.T) + lambd / m * W2
    ### END CODE HERE ###
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW1 = 1./m * np.dot(dZ1, X.T) + lambd / m * W1
    ### END CODE HERE ###
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients


# 使用 dropout 正则化
def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    实现使用 dropout 正则化的前向传播: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    :param X: 输入数据集, of shape (2, number of examples)
    :param parameters: python dictionary，包含参数： "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    :param keep_prob: drop-out（随即删除）过程中保留一个神经元的概率
    
    :return A3: 最后一层的激活值, 前向传播的输出, of shape (1,1)
    :return cache: tuple, 存储着用来计算后向传播的信息
    """
    
    np.random.seed(1)
    
    # 重新取出参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    # 随即删除节点（神经元）
    ### START CODE HERE ### (approx. 4 lines)         
    D1 = np.random.rand(A1.shape[0], A1.shape[1])    # Step 1: 初始化矩阵 D （D 与 A 的维度相同，D 为 1 则保留相应的神经元，为 0 则删除）
    D1 = (D1 < keep_prob).astype(int)                 # Step 2: 将 D1 的值转化为 0 or 1 (使用 keep_prob 作为阈值)
    A1 = A1 * D1                                      # Step 3: 删除 A1 的一些节点
    A1 = A1 / keep_prob                               # Step 4: 将未删除的神经元的值缩放
    ### END CODE HERE ###

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    ### START CODE HERE ### (approx. 4 lines)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])    # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    D2 = (D2 < keep_prob).astype(int)                 # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    A2 = A2 * D2                                      # Step 3: shut down some neurons of A2
    A2 = A2 / keep_prob                               # Step 4: scale the value of neurons that haven't been shut down
    ### END CODE HERE ###

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache


def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    实现使用 dropout 正则化的后向传播
    
    :param X: 输入数据集, of shape (2, number of examples)
    :param Y: 正确的标签向量, of shape (output size, number of examples)
    :param cache: forward_propagation_with_dropout() 输出的 cache
    :param keep_prob: drop-out（随即删除）过程中保留一个神经元的概率
    
    :return gradients: 包含了每个参数W/b、激活值A、预激活值Z的梯度的字典
    """
    
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)
    ### START CODE HERE ### (≈ 2 lines of code)
    dA2 = dA2 * D2           # Step 1: 使用 mask D2 来删除正向传播中删除的那些神经元
    dA2 = dA2 / keep_prob    # Step 2: 缩放未删除的神经元的值
    ### END CODE HERE ###
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    ### START CODE HERE ### (≈ 2 lines of code)
    dA1 = dA1 * D1           # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    dA1 = dA1 / keep_prob    # Step 2: Scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients


def main():
    # 导入数据，并绘制红蓝点分布图
    train_X, train_Y, test_X, test_Y = load_2D_dataset()    # load_2D_dataset()函数中将 c=train_Y 修改为 c=np.squeeze(train_Y)

    # 使用 dropout 正则化

    # 使用带 dropout 正则化的神经网络模型训练参数
    parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)

    # 预测并打印准确率
    print ("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    # 绘制决策边界图
    plt.title("Model with dropout")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

    """
    # 使用 L2 正则化

    # 使用带 L2 正则化的神经网络模型训练参数
    parameters = model(train_X, train_Y, lambd = 0.7)

    print ("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    # 绘制决策边界图
    plt.title("Model with L2-regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    """


if __name__ == '__main__':
    main()