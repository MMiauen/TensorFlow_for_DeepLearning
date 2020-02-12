# 第三章 TensorFlow基本开发步骤——以逻辑回归拟合二维数据为例
## 3.1实例1 从一组看似混乱的数据中找出y≈2x的规律
具体描述：假设有一组数据集，x和y的对应关系是y≈2x，我们希望能让神经网络学习这些样本，并找到这条规律。深度学习中，大概有如下四个步骤：

（1）准备数据→（2）搭建模型→（3）迭代训练→（4）使用模型
### 3.1.1数据准备


```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# x,y
train_X=np.linspace(-1,1,100)
train_Y=2*train_x+np.random.randn(*train_x.shape)*0.3 # 在y=2x的基础上加上噪声，实现“约等于”
#显式模拟数据点
plt.plot(train_X,train_Y,'ro',label='Original data')
plt.legend()
plt.show()
```


![png](output_1_0.png)


### 3.1.2 搭建模型
模型根据数据流动可分为正向模型和反向模型：通过正向模型生成一个值，然后观察其与真实值的差距，再通过反向过程进行参数调整


```python
#占位符
X=tf.placeholder('float')
Y=tf.placeholder('float')
#模型参数
W=tf.Variable(tf.random_normal([1]),name='weight') # W初始化为[-1,1]间的随机数，形状为一维
b=tf.Variable(tf.zeros([1]),name='bias') # b初始化为0，形状为一维
#前向结构
z=tf.multiply(X,W)+b
#反向优化
cost=tf.reduce_mean(tf.square(Y-z))
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #梯度下降
```

### 3.1.3 迭代训练模型
TensorFlow中的任务是通过session来进行的


```python
#后续作图用到的函数
plotdata={"batchsize":[],'loss':[]}
def moving_average(a,w=10):
        if len(a)<w:
            return a[:]
        return [val if idx < w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]

###开始####
#初始化所有变量
init=tf.global_variables_initializer()
#设置训练迭代次数
training_epochs=20
display_step=2
#启动session
with tf.Session() as sess:
    sess.run(init)
    plotdata={"batchsize":[],'loss':[]}  #存放批次值和损失值
    #向模型输入数据
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):  
            sess.run(optimizer,feed_dict={X:x,Y:y})
            
        if epoch%display_step==0:
            loss=sess.run(cost,feed_dict={X:train_x,Y:train_y})
            print("Epoch",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
            if not(loss=="NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    print("Finished!")
    print("cost=",sess.run(cost,feed_dict={X:train_x,Y:train_y}),"W=",sess.run(W),"b=",sess.run(b))

    #模型可视化
    plt.plot(train_X,train_Y,'ro',label='Original data')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted line')
    plt.legend()
    plt.show()

    plotdata['avgloss']=moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"],plotdata["avgloss"],'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title("Minibatch run vs. Training loss")
    plt.show()
    
    #使用模型
    print("使用模型")
    print("x=1,y=",sess.run(z,feed_dict={X:1}))
```

    Epoch 1 cost= 0.07318244 W= [1.8917329] b= [0.0411104]
    Epoch 3 cost= 0.06481901 W= [1.9747607] b= [0.0198762]
    Epoch 5 cost= 0.06328333 W= [1.9973855] b= [0.01137286]
    Epoch 7 cost= 0.06296215 W= [2.003255] b= [0.00912356]
    Epoch 9 cost= 0.06288451 W= [2.0047727] b= [0.00854117]
    Epoch 11 cost= 0.062864825 W= [2.0051644] b= [0.00839077]
    Epoch 13 cost= 0.0628597 W= [2.005267] b= [0.00835149]
    Epoch 15 cost= 0.06285845 W= [2.005292] b= [0.00834183]
    Epoch 17 cost= 0.06285813 W= [2.0052986] b= [0.0083394]
    Epoch 19 cost= 0.06285803 W= [2.0053003] b= [0.00833872]
    Finished!
    cost= 0.06285803 W= [2.0053005] b= [0.00833863]
    


![png](output_5_1.png)



![png](output_5_2.png)


    使用模型
    x=1,y= [2.0136392]
    

## 3.3 TensorFlow开发的基本步骤
### step1.定义输入结点
占位符定义（最常用）

字典定义（输入结点较多时使用）

直接定义（很少使用，因为会使模型通用性变差）

### step2.定义“学习参数”的变量
这部分与步骤1很像，分为直接定义和字典定义两种方式

### step3.定义“运算”
定义正向传播模型、定义损失函数

### step4.优化函数、优化目标

### step5.初始化所有变量
初始化所有变量的过程，虽然只有一句代码但是也非常关键。session创建好之后，第一件事就是初始化。

init=tf.global_variables_initializer()

with tf.Session() as sess:

   sess.run(init)
   
必须在所有变量定义完之后，再使用global_variables_initializer函数对其进行初始化，否则无法用session中的run来进行算值

### step6.迭代更新到最优解
### step7.测试模型
### step8.使用模型
通常搭建好的模型不直接使用，而是先保存起来，再通过载入已有模型进行实际的使用。关于模型的载入和读取将在后续章节介绍。
   
   


