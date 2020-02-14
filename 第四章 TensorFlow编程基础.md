# 4.1 编程模型
## 4.1.1 模型运行机制
TensorFlow的运行机制属于“定义”与“运行”相分离。即分为：模型构建和模型运行 

“定义”→在一个“图”容器中完成，一个图代表一个计算任务

“运行”→“图”在会话（session）中被启动
## 4.1.2 实例：编写hello world程序演示session的使用


```python
import tensorflow as tf
hello=tf.constant("hello,tensorflow") #tf.constant是定义常量
sess=tf.Session()
print(sess.run(hello))
sess.close()
```

    b'hello,tensorflow'
    


```python
import tensorflow as tf 
a=tf.constant(3)
b=tf.constant(4)
with tf.Session() as sess:
    print("a+b=",sess.run(a+b))
    print("a*b=",sess.run(a*b))
```

    a+b= 7
    a*b= 12
    

## 4.1.3 演示注入机制（feed）


```python
import tensorflow as tf
a=tf.placeholder(tf.int16)
b=tf.placeholder(tf.int16)
add=tf.add(a,b)
mul=tf.multiply(a,b)

with tf.Session() as sess:
    print("a+b=",sess.run(add,feed_dict={a:3,b:4}))
    print("a*b=",sess.run(mul,feed_dict={a:3,b:4}))
    #使用注入机制获取结点
    print(sess.run([add,mul],feed_dict={a:3,b:4}))
```

    a+b= 7
    a*b= 12
    [7, 12]
    

## 4.1.4 保存和载入模型
保存：需要先建立一个saver，然后在session中通过调用saver的save即可将模型保存下来

载入：在session中通过调用saver的restore（）函数从指定路径找到模型文件，并覆盖到相关参数中


```python
##保存模型

##之前是各种构建模型graph的操作
saver=tf.train.Saver()                           #生成saver
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  #对模型进行初始化
    #数据丢入模型训练blablabla
    #训练完之后使用saver.save来保存
    saver.save(sess,"save_path/file_name")
    
##载入模型

saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"save_path/file_name")
```

## 4.1.5 实例：保存/载入线性回归模型


```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# x,y
train_X=np.linspace(-1,1,100)
train_Y=2*train_X+np.random.randn(*train_X.shape)*0.3 # 在y=2x的基础上加上噪声，实现“约等于”
#显式模拟数据点
plt.plot(train_X,train_Y,'ro',label='Original data')
plt.legend()
plt.show()

#重置图
tf.reset_default_graph()

####初始化等操作####
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
#后续作图用到的函数
plotdata={"batchsize":[],'loss':[]}
def moving_average(a,w=10):
        if len(a)<w:
            return a[:]
        return [val if idx < w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]

#初始化所有变量
init=tf.global_variables_initializer()
#设置训练迭代次数
training_epochs=20
display_step=2


####保存操作##
saver=tf.train.Saver()
# savedir="H:/DeepLearning/BOOK_TensorFlowForDeepLearning"

###启动session###
with tf.Session() as sess:
    sess.run(init)
    plotdata={"batchsize":[],'loss':[]}  #存放批次值和损失值
    #向模型输入数据
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):  
            sess.run(optimizer,feed_dict={X:x,Y:y})
            
        if epoch%display_step==0:
            loss=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print("Epoch",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
            if not(loss=="NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    print("Finished!")
    
    saver.save(sess,"H:/DeepLearning/BOOK_TensorFlowForDeepLearning/Unit4_LinerModel/LinerModel.cpkt")  # 保存
    
    print("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_Y}),"W=",sess.run(W),"b=",sess.run(b))
```


    <Figure size 640x480 with 1 Axes>


    Epoch 1 cost= 0.30416298 W= [1.3110032] b= [0.21125937]
    Epoch 3 cost= 0.09243593 W= [1.8552046] b= [0.06459747]
    Epoch 5 cost= 0.076264165 W= [2.002681] b= [0.0090504]
    Epoch 7 cost= 0.075490534 W= [2.0409281] b= [-0.0056092]
    Epoch 9 cost= 0.07552233 W= [2.0508199] b= [-0.00940479]
    Epoch 11 cost= 0.075546056 W= [2.0533772] b= [-0.01038621]
    Epoch 13 cost= 0.07555322 W= [2.0540378] b= [-0.01063969]
    Epoch 15 cost= 0.075555146 W= [2.0542092] b= [-0.01070548]
    Epoch 17 cost= 0.07555565 W= [2.054253] b= [-0.01072224]
    Epoch 19 cost= 0.07555579 W= [2.0542653] b= [-0.01072693]
    Finished!
    cost= 0.07555581 W= [2.054267] b= [-0.0107277]
    


```python
# 载入模型
import tensorflow as tf
with tf.Session() as sess2:
    saver.restore(sess2,"H:\\DeepLearning\\BOOK_TensorFlowForDeepLearning\\Unit4_LinerModel\LinerModel.cpkt")
    print("x=2,z=",sess2.run(z,feed_dict={X:2}))
```

    INFO:tensorflow:Restoring parameters from H:\DeepLearning\BOOK_TensorFlowForDeepLearning\Unit4_LinerModel\LinerModel.cpkt
    x=2,z= [4.097806]
    

## 4.1.6 分析模型内容
虽然模型已经保存下来了，并实现了读取。但仍然对我们不透明，我们希望通过编写代码把模型中的内容打印出来，看看到底保存了些啥


```python
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file("H:\\DeepLearning\\BOOK_TensorFlowForDeepLearning\\Unit4_LinerModel\LinerModel.cpkt",None,True)
```

    tensor_name:  bias
    [-0.0107277]
    tensor_name:  weight
    [2.054267]
    # Total number of params: 2
    

## 4.1.7 TensorBoard可视化
可以通过网页来观察模型的结构和训练过程中各个参数的变化。以线性回归模型为例：


```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# x,y
train_X=np.linspace(-1,1,100)
train_Y=2*train_X+np.random.randn(*train_X.shape)*0.3 # 在y=2x的基础上加上噪声，实现“约等于”
#显式模拟数据点
plt.plot(train_X,train_Y,'ro',label='Original data')
plt.legend()
plt.show()

#重置图
tf.reset_default_graph()

####初始化等操作####
#占位符
X=tf.placeholder('float')
Y=tf.placeholder('float')
#模型参数
W=tf.Variable(tf.random_normal([1]),name='weight') # W初始化为[-1,1]间的随机数，形状为一维
b=tf.Variable(tf.zeros([1]),name='bias') # b初始化为0，形状为一维
#前向结构
z=tf.multiply(X,W)+b
tf.summary.histogram('z',z)
#反向优化
cost=tf.reduce_mean(tf.square(Y-z))
tf.summary.scalar('loss_function',cost)
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #梯度下降
#后续作图用到的函数
plotdata={"batchsize":[],'loss':[]}
def moving_average(a,w=10):
        if len(a)<w:
            return a[:]
        return [val if idx < w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]

#初始化所有变量
init=tf.global_variables_initializer()
#设置训练迭代次数
training_epochs=20
display_step=2


####保存操作##
saver=tf.train.Saver()
# savedir="H:/DeepLearning/BOOK_TensorFlowForDeepLearning"

###启动session###
with tf.Session() as sess:
    sess.run(init)
    plotdata={"batchsize":[],'loss':[]}  #存放批次值和损失值
    
    
    merged_summary_op=tf.summary.merge_all()
    summary_writer=tf.summary.FileWriter('H:\\DeepLearning\\BOOK_TensorFlowForDeepLearning\\Unit4_LinerModel\\mnist_with_summaries',sess.graph)
    
    
    
    #向模型输入数据
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):  
            sess.run(optimizer,feed_dict={X:x,Y:y})
            
            
    summary_str=sess.run(merged_summary_op,feed_dict={X:x,Y:y})
    summary_writer.add_summary(summary_str,epoch)
    
            
#         if epoch%display_step==0:
#             loss=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
#             print("Epoch",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
#             if not(loss=="NA"):
#                 plotdata["batchsize"].append(epoch)
#                 plotdata["loss"].append(loss)
#     print("Finished!")
    
    saver.save(sess,"H:/DeepLearning/BOOK_TensorFlowForDeepLearning/Unit4_LinerModel/LinerModel.cpkt")  # 保存
    
    print("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_Y}),"W=",sess.run(W),"b=",sess.run(b))
```


![png](output_13_0.png)


    cost= 0.07032787 W= [2.0324483] b= [-0.02592937]
    
