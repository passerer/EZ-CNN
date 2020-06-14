## EZ-CNN

+ 基于C++11的卷积神经网络实现, 无外部依赖库.

+ 无gpu, 无多线程, 无SSE加速. 纯粹用于学习.

## Structure

+ 类似caffe v1，以层为单位，包含卷积层、池化层、全连接层、softmax层、reshape层、loss层等.

+ 主要数据用一个名为"Tensor"的结构体表示.

+ 计算基于下标的形式而非矩阵形式

## Thinking

+ 基于下标的计算容易出bug，以后会考虑将数据用mat表示

+ 有时间会加入bn层、带孔卷积、dropout层等常用层
