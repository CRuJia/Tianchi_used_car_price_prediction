## 1  内容介绍

模型融合是比赛后期一个重要的环节，大体来说有如下的类型方式。

1.1  简单加权融合:
    - 回归（分类概率）：算术平均融合（Arithmetic mean），几何平均融合（Geometric mean）；
    - 分类：投票（Voting)
    - 综合：排序融合(Rank averaging)，log融合


1.2  stacking/blending:
    - 构建多层模型，并利用预测结果再拟合预测。


1.3  boosting/bagging（在xgboost，Adaboost,GBDT中已经用到）:
    - 多树的提升方法

# 2 Stacking相关理论介绍
#### 1)  什么是 stacking

简单来说 stacking 就是当用初始训练数据学习出若干个基学习器后，将这几个学习器的预测结果作为新的训练集，来学习一个新的学习器。
![](http://jupter-oss.oss-cn-hangzhou.aliyuncs.com/public/files/image/2326541042/1584448793231_6TygjXwjNb.jpg)
将个体学习器结合在一起的时候使用的方法叫做结合策略。对于分类问题，我们可以使用投票法来选择输出最多的类。对于回归问题，我们可以将分类器输出的结果求平均值。

上面说的投票法和平均法都是很有效的结合策略，还有一种结合策略是使用另外一个机器学习算法来将个体机器学习器的结果结合在一起，这个方法就是Stacking。

在stacking方法中，我们把个体学习器叫做初级学习器，用于结合的学习器叫做次级学习器或元学习器（meta-learner），次级学习器用于训练的数据叫做次级训练集。次级训练集是在训练集上用初级学习器得到的。
#### 2)  如何进行 stacking
算法示意图如下：
![](http://jupter-oss.oss-cn-hangzhou.aliyuncs.com/public/files/image/2326541042/1584448806789_1ElRtHaacw.jpg)

> 引用自 西瓜书《机器学习》
* 过程1-3 是训练出来个体学习器，也就是初级学习器。
* 过程5-9是 使用训练出来的个体学习器来得预测的结果，这个预测的结果当做次级学习器的训练集。
* 过程11 是用初级学习器预测的结果训练出次级学习器，得到我们最后训练的模型。

 #### 3）Stacking的方法讲解

首先，我们先从一种“不那么正确”但是容易懂的Stacking方法讲起。

Stacking模型本质上是一种分层的结构，这里简单起见，只分析二级Stacking.假设我们有2个基模型 Model1_1、Model1_2 和 一个次级模型Model2

**Step 1.** 基模型 Model1_1，对训练集train训练，然后用于预测 train 和 test 的标签列，分别是P1，T1

Model1_1 模型训练:

$$
\left(\begin{array}{c}{\vdots} \\ {X_{train}} \\ {\vdots}\end{array}\right) \overbrace{\Longrightarrow}^{\text {Model1_1 Train} }\left(\begin{array}{c}{\vdots} \\ {Y}_{True} \\ {\vdots}\end{array}\right)
$$

训练后的模型 Model1_1 分别在 train 和 test 上预测，得到预测标签分别是P1，T1

$$
\left(\begin{array}{c}{\vdots} \\ {X_{train}} \\ {\vdots}\end{array}\right) \overbrace{\Longrightarrow}^{\text {Model1_1 Predict} }\left(\begin{array}{c}{\vdots} \\ {P}_{1} \\ {\vdots}\end{array}\right)
$$

$$
\left(\begin{array}{c}{\vdots} \\ {X_{test}} \\ {\vdots}\end{array}\right) \overbrace{\Longrightarrow}^{\text {Model1_1 Predict} }\left(\begin{array}{c}{\vdots} \\ {T_{1}} \\ {\vdots}\end{array}\right)
$$

**Step 2.** 基模型 Model1_2 ，对训练集train训练，然后用于预测train和test的标签列，分别是P2，T2

Model1_2 模型训练:

$$
\left(\begin{array}{c}{\vdots} \\ {X_{train}} \\ {\vdots}\end{array}\right) \overbrace{\Longrightarrow}^{\text {Model1_2 Train} }\left(\begin{array}{c}{\vdots} \\ {Y}_{True} \\ {\vdots}\end{array}\right)
$$

训练后的模型 Model1_2 分别在 train 和 test 上预测，得到预测标签分别是P2，T2

$$
\left(\begin{array}{c}{\vdots} \\ {X_{train}} \\ {\vdots}\end{array}\right) \overbrace{\Longrightarrow}^{\text {Model1_2 Predict} }\left(\begin{array}{c}{\vdots} \\ {P}_{2} \\ {\vdots}\end{array}\right)
$$

$$
\left(\begin{array}{c}{\vdots} \\ {X_{test}} \\ {\vdots}\end{array}\right) \overbrace{\Longrightarrow}^{\text {Model1_2 Predict} }\left(\begin{array}{c}{\vdots} \\ {T_{2}} \\ {\vdots}\end{array}\right)
$$

**Step 3.** 分别把P1,P2以及T1,T2合并，得到一个新的训练集和测试集train2,test2.

$$
\overbrace{\left(\begin{array}{c}{\vdots} \\ {P_{1}} \\ {\vdots}\end{array} \begin{array}{c}{\vdots} \\ {P_{2}} \\ {\vdots}\end{array} \right)}^{\text {Train_2 }}  
and 
\overbrace{\left(\begin{array}{c}{\vdots} \\ {T_{1}} \\ {\vdots}\end{array} \begin{array}{c}{\vdots} \\ {T_{2}} \\ {\vdots}\end{array} \right)}^{\text {Test_2 }}
$$

再用 次级模型 Model2 以真实训练集标签为标签训练,以train2为特征进行训练，预测test2,得到最终的测试集预测的标签列 $Y_{Pre}$。

$$
\overbrace{\left(\begin{array}{c}{\vdots} \\ {P_{1}} \\ {\vdots}\end{array} \begin{array}{c}{\vdots} \\ {P_{2}} \\ {\vdots}\end{array} \right)}^{\text {Train_2 }} \overbrace{\Longrightarrow}^{\text {Model2 Train} }\left(\begin{array}{c}{\vdots} \\ {Y}_{True} \\ {\vdots}\end{array}\right)
$$

$$
\overbrace{\left(\begin{array}{c}{\vdots} \\ {T_{1}} \\ {\vdots}\end{array} \begin{array}{c}{\vdots} \\ {T_{2}} \\ {\vdots}\end{array} \right)}^{\text {Test_2 }} \overbrace{\Longrightarrow}^{\text {Model1_2 Predict} }\left(\begin{array}{c}{\vdots} \\ {Y}_{Pre} \\ {\vdots}\end{array}\right)
$$

这就是我们两层堆叠的一种基本的原始思路想法。在不同模型预测的结果基础上再加一层模型，进行再训练，从而得到模型最终的预测。

Stacking本质上就是这么直接的思路，但是直接这样有时对于如果训练集和测试集分布不那么一致的情况下是有一点问题的，其问题在于用初始模型训练的标签再利用真实标签进行再训练，毫无疑问会导致一定的模型过拟合训练集，这样或许模型在测试集上的泛化能力或者说效果会有一定的下降，因此现在的问题变成了如何降低再训练的过拟合性，这里我们一般有两种方法。
* 1. 次级模型尽量选择简单的线性模型
* 2. 利用K折交叉验证

K-折交叉验证：
训练：

![](http://jupter-oss.oss-cn-hangzhou.aliyuncs.com/public/files/image/2326541042/1584448819632_YvJOXMk02P.jpg)

预测：

![](http://jupter-oss.oss-cn-hangzhou.aliyuncs.com/public/files/image/2326541042/1584448826203_k8KPy9n7D9.jpg)
