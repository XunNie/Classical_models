## Lenet介绍-使用Tensorflow实现

### 模型介绍
LeNet5诞生于1994年，由Yann LeCun提出，充分考虑图像的相关性。当时结构的特点如下：
1）每个卷积层包含三个部分：卷积（Conv）、池化（ave-pooling）、非线性激活函数（sigmoid）
2）MLP作为最终的分类器
3）层与层之间稀疏连接减少计算复杂度
## 网络结构介绍：
Input Layer：1\*32\*32图像
Conv1 Layer：包含6个卷积核，kernal size：5\*5，parameters:(5\*5+1)\*6=156个
Subsampling Layer 1：average pooling，size：2*2
                                  Activation Function：sigmoid
Conv3 Layer：包含16个卷积核，kernal size：5*5  ->16个Feature Map
Subsampling Layer 1：average pooling，size：2*2
Conv5 Layer：包含120个卷积核，kernal size：5*5
Fully Connected Layer：Activation Function：sigmoid
Output Layer：Gaussian connection

## 网络模型图-TensorBoard
![这里写图片描述](https://img-blog.csdn.net/2018091314324314?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTQ2OTI3Mg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
## 结构图实现源码地址
[https://github.com/XunNie/Classical_models/blob/master/Lenet-5.py](https://github.com/XunNie/Classical_models/blob/master/Lenet-5.py)

## 参考文献
[https://blog.csdn.net/xjy104165/article/details/78218057](https://blog.csdn.net/xjy104165/article/details/78218057)
[https://blog.csdn.net/roguesir/article/details/73770448](https://blog.csdn.net/roguesir/article/details/73770448)
