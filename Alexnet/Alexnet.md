## AlexNet 介绍及Tensorflow实现源码
AlexNet是神经网络之父Hinton的学生Alex Krizhevsky开发完成，它总共有8层，其中有5个卷积层，3个全链层。
### 网络结构介绍
介绍内容[参考链接](https://blog.csdn.net/taoyanqi8932/article/details/71081390)

第一个卷积层

    输入的图片大小为:224*224*3

    第一个卷积层为:11*11*96即尺寸为11*11,有96个卷积核,步长为4,卷积层后跟ReLU,因此输出的尺寸为 224/4=56,去掉边缘为55,因此其输出的每个feature map 为 55*55*96,同时后面跟LRN层,尺寸不变.

    最大池化层,核大小为3*3,步长为2,因此feature map的大小为:27*27*96.

第二层卷积层

    输入的tensor为27*27*96

    卷积和的大小为: 5*5*256,步长为1,尺寸不会改变,同样紧跟ReLU,和LRN层.

    最大池化层,和大小为3*3,步长为2,因此feature map为:13*13*256

第三层至第五层卷积层

    输入的tensor为13*13*256

    第三层卷积为 3*3*384,步长为1,加上ReLU

    第四层卷积为 3*3*384,步长为1,加上ReLU

    第五层卷积为 3*3*256,步长为1,加上ReLU

    第五层后跟最大池化层,核大小3*3,步长为2,因此feature map:6*6*256

第六层至第八层全连接层

接下来的三层为全连接层,分别为:
1. FC : 4096 + ReLU
2. FC:4096 + ReLU
3. FC: 1000
最后一层为softmax为1000类的概率值.
2. AlexNet中的trick

AlexNet将CNN用到了更深更宽的网络中,其效果分类的精度更高相比于以前的LeNet,其中有一些trick是必须要知道的.
ReLU的应用

AlexNet使用ReLU代替了Sigmoid,其能更快的训练,同时解决sigmoid在训练较深的网络中出现的梯度消失,或者说梯度弥散的问题.
Dropout随机失活

随机忽略一些神经元,以避免过拟合,
重叠的最大池化层

在以前的CNN中普遍使用平均池化层,AlexNet全部使用最大池化层,避免了平均池化层的模糊化的效果,并且步长比池化的核的尺寸小,这样池化层的输出之间有重叠,提升了特征的丰富性.
提出了LRN层

局部响应归一化,对局部神经元创建了竞争的机制,使得其中响应小打的值变得更大,并抑制反馈较小的.
使用了GPU加速计算
使用了gpu加速神经网络的训练
数据增强
使用数据增强的方法缓解过拟合现象.
### 网络结构图
![](https://img-blog.csdn.net/20170502145806536?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGFveWFucWk4OTMy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### Tensorflow 实现源码地址
[https://github.com/XunNie/Classical_models/blob/master/Alexnet/Alexnet.py](https://github.com/XunNie/Classical_models/blob/master/Alexnet/Alexnet.py)

### Tensorboard 显示图
![这里写图片描述](https://img-blog.csdn.net/20180913172913559?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTQ2OTI3Mg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
### Alexnet特点
1. 相比Lenet提升了图像分类的准确度
2. 使用LRN(局部相应标准化)，跟dropout功能，很少被使用，dropout使用更为频繁。减少过拟合的问题。
3. Lenet使用的激励函数为sigmoid，因此训练比较慢，而Alexnet使用的是Relu函数。减少梯度消失的问题
### 参考文献
[https://blog.csdn.net/taoyanqi8932/article/details/71081390](https://blog.csdn.net/taoyanqi8932/article/details/71081390)
