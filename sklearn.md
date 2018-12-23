### 一、特征工程

sklearn特征抽取API ：sklearn.feature_extraction 

1.对字典数据进行特征值化 ：sklearn.feature_extraction.DictVectorizer 

默认 DictVectorizer(sparse=True,…) 

DictVectorizer.fit_transform(X)  ==》  X:字典或者包含字典的迭代器   返回值：返回sparse矩阵

DictVectorizer.inverse_transform(X) ==》X:array数组或者sparse矩阵  返回值:转换之前数据格式

DictVectorizer.get_feature_names() ==》返回类别名称

DictVectorizer.transform(X) ==》按照原先的标准转换

```python
from sklearn.feature_extraction import DictVectorizer

def dictvec():
    """
    字典数据抽取
    :return: None
    """
    # 实例化
    dict = DictVectorizer(sparse=False)

    # 调用fit_transform
    data = dict.fit_transform([{'city': '北京','temperature': 100}, {'city': '上海','temperature':60}, {'city': '深圳','temperature': 30}])

    print(dict.get_feature_names())
    print(dict.inverse_transform(data))
    print(data)
```

2.对文本数据进行特征值化：  sklearn.feature_extraction.text.CountVectorizer 

CountVectorizer.fit_transform(X,y)  ==》X:文本或者包含文本字符串的可迭代对象  返回值：返回sparse矩阵

CountVectorizer.inverse_transform(X) ==》X:array数组或者sparse矩阵  返回值:转换之前数据格式

CountVectorizer.get_feature_names() ==》返回值:单词列表

```python
from sklearn.feature_extraction.text import CountVectorizer

# 普通英文文本分析（一般不用），默认按空格分词
def countvec():
    """
    对文本进行特征值化
    :return: None
    """
    cv = CountVectorizer()

    data = cv.fit_transform(["人生 苦短，我 喜欢 python", "人生漫长，不用 python"])

    print(cv.get_feature_names())
    print(data.toarray())
```

```python
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# pip install jieba   分词工具
# 可用于中文进行文本分析
def cutword():
    # jieba分词
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")
    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    # 转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)
    # 吧列表转换成字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)
    return c1, c2, c3


def hanzivec():
    """
    中文特征值化
    :return: None
    """
    c1, c2, c3 = cutword()
    print(c1, c2, c3)
    cv = CountVectorizer()
    data = cv.fit_transform([c1, c2, c3])
    print(cv.get_feature_names())
    print(data.toarray())
    
    
# Tf:term frequency:词的频率
# idf:逆⽂档频率inverse document frequency 
def tfidfvec():
    """
    中文特征值化，一般使用这个
    :return: None
    """
    c1, c2, c3 = cutword()

    tf = TfidfVectorizer()    
    data = tf.fit_transform([c1, c2, c3])
    print(tf.get_feature_names())
    print(data.toarray())
```

3.特征处理

数值型数据：标准缩放   1、归一化  2、标准化   3、缺失值

类别型数据：one-hot编码

时间类型：时间的切分

sklearn特征处理API ： sklearn. preprocessing  

```python
# 归一化：通过对原始数据进行变换把数据映射到(默认为[0,1])之间
from sklearn.preprocessing import MinMaxScaler

def mm():
    """
    归一化处理
    :return: NOne
    """
    mm = MinMaxScaler(feature_range=(2, 3))
    data = mm.fit_transform([[90,2,10,40],[60,4,15,45],[75,3,13,46]])
    print(data)
    
# 注意：最大值与最小值非常容易受异常点影响，所以这种方法鲁棒性较差，只适合传统精确小数据场景
```

```python
# 标准化: 通过对原始数据进行变换把数据变换到均值为0,标准差为1范围内
from sklearn.preprocessing import StandardScaler

def stand():
    """
    标准化缩放
    :return:
    """
    std = StandardScaler()
    data = std.fit_transform([[ 1., -1., 3.],[ 2., 4., 2.],[ 4., 6., -1.]])
    print(data)
```

```python
# 缺失值处理：一般在数据分析过程就要处理好
# 方法：删除（整行或整列）、插补（平均值或中位数等）
from sklearn.preprocessing import Imputer

def im():
    """
    缺失值处理
    """
    # NaN, nan
    im = Imputer(missing_values='NaN', strategy='mean', axis=0)
    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])
    # data：转换后的形状相同的array
    print(data)
```

4.特征选择

特征选择就是单纯地从提取到的所有特征中选择部分特征作为训练集特征 

主要方法（三大武器）：Filter(过滤式):VarianceThreshold

​                      			  Embedded(嵌入式)：正则化、决策树

​                                          Wrapper(包裹式)

```python
from sklearn.feature_selection import VarianceThreshold

def var():
    """
    特征选择-删除低方差的特征，给定一个阈值
    """
    var = VarianceThreshold(threshold=1.0)
    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print(data)
```

5.sklearn降维

sklearn降维API ：sklearn. decomposition

PCA是一种分析、简化数据集的技术   可以削减回归分析或者聚类分析中特征的数量 

```python
from sklearn.decomposition import PCA

def pca():
    """
    主成分分析进行特征降维，参数：选择保留原数据多少信息
    """
    pca = PCA(n_components=0.9)
    data = pca.fit_transform([[2,8,4,5],[6,3,0,8],[5,4,9,1]])
    print(data)
```



### 二、机器学习

监督学习（输入数据有特征有标签 ）

​	分类(目标值离散型 )    k-近邻算法、贝叶斯分类、决策树与随机森林、逻辑回归、神经网络

​	回归 (目标值连续型 )   线性回归、岭回归

​	标注    隐马尔可夫模型     (不做要求)

无监督学习（输入数据有特征无标签）

​	聚类    k-means

机器学习开发流程（重要）：
	即 建立模型：根据数据类型划分应用种类  （模型：算法 + 数据）
	1、原始数据明确问题做什么
	2、数据的基本处理：pd去处理数据（缺失值，合并表。。。。。）
	3、特征工程 （特征进行处理） （重要）
	4、找到合适算法去进行预测
	（根据问题和数据看是用 分类 还是 回归 ）
	5、模型的评估，判定效果
    	   合格：   ---》  上线使用以API形式提供
	  不合格： ---》 重复步骤3和步骤4，即换算法或处理特征工程	

1.数据集划分

sklearn数据集划分API ：sklearn.model_selection.train_test_split 

x_train, x_test, y_train, y_test = train_test_split(lr.data, lr.target, test_size=0.25)
从左到右依次为：训练集数据、测试集数据、训练集目标值、测试集目标值

2.分类算法-k近邻算法 

一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别 

使用欧式距离计算：
两个样本a(a1,a2,a3),b(b1,b2,b3)：其欧式距离为对应点的差值的平方和再开根号

api：sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='auto') 

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
# GridSearchCV  网格搜索  调参使用

def knncls():
    """
    K-近邻预测用户签到位置
    :return:None
    """
    # 读取数据
    data = pd.read_csv("./data/FBlocation/train.csv")

    # 处理数据
    # 1、缩小数据,查询数据晒讯
    data = data.query("x > 1.0 &  x < 1.25 & y > 2.5 & y < 2.75")

    # 处理时间的数据
    time_value = pd.to_datetime(data['time'], unit='s')

    print(time_value)

    # 把日期格式转换成 字典格式
    time_value = pd.DatetimeIndex(time_value)

    # 构造一些特征
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday

    # 把时间戳特征删除
    data = data.drop(['time'], axis=1)

    print(data)

    # 把签到数量少于n个目标位置删除
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 3].reset_index()
    data = data[data['place_id'].isin(tf.place_id)]

    # 取出数据当中的特征值和目标值
    y = data['place_id']
    x = data.drop(['place_id'], axis=1)

    # 进行数据的分割训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程（标准化）
    std = StandardScaler()

    # 对测试集和训练集的特征值进行标准化
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 进行算法流程 # 超参数
    knn = KNeighborsClassifier()

    # # fit， predict,score
    # knn.fit(x_train, y_train)
    #
    # # 得出预测结果
    # y_predict = knn.predict(x_test)
    #
    # print("预测的目标签到位置为：", y_predict)
    #
    # # 得出准确率
    # print("预测的准确率:", knn.score(x_test, y_test))

    # 构造一些参数的值进行搜索
    param = {"n_neighbors": [3, 5, 10]}

    # 进行网格搜索
    gc = GridSearchCV(knn, param_grid=param, cv=2)
    gc.fit(x_train, y_train)

    # 预测准确率
    print("在测试集上准确率：", gc.score(x_test, y_test))
    print("在交叉验证当中最好的结果：", gc.best_score_)
    print("选择最好的模型是：", gc.best_estimator_)
    print("每个超参数每次交叉验证的结果：", gc.cv_results_)
```

2.分类模型的评估

estimator.score()

一般最常见使用的是准确率，即预测结果正确的百分比

精确率(Precision)与召回率(Recall) 

精确率：预测结果为正例样本中真实为正例的比例（查得准） 

召回率：真实为正例的样本中预测结果为正例的比例（查的全，对正样本的区分能力）

分类模型评估API ：sklearn.metrics.classification_report 

sklearn.metrics.classification_report(y_true, y_pred, target_names=None)

y_true：真实目标值          y_pred：估计器预测目标值       target_names：目标类别名称

return：每个类别精确率与召回率

3.分类算法-朴素贝叶斯 

sklearn朴素贝叶斯实现API ：sklearn.naive_bayes.MultinomialNB 

优点：

​	朴素贝叶斯模型发源于古典数学理论，有稳定的分类效率。

​	对缺失数据不太敏感，算法也比较简单，常用于文本分类。

​	分类准确度高，速度快

•缺点：

​	需要知道先验概率P(F1,F2,…|C)，因此在某些时候会由于假设的先验模型的原因导致预测效果不佳

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def naviebayes():
    """
    朴素贝叶斯进行文本分类
    :return: None
    """
    news = fetch_20newsgroups(subset='all')

    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 对数据集进行特征抽取
    tf = TfidfVectorizer()

    # 以训练集当中的词的列表进行每篇文章重要性统计['a','b','c','d']
    x_train = tf.fit_transform(x_train)
    print(tf.get_feature_names())
    x_test = tf.transform(x_test)

    # 进行朴素贝叶斯算法的预测
    mlt = MultinomialNB(alpha=1.0)   # alpha：拉普拉斯平滑系数
    print(x_train.toarray())
    mlt.fit(x_train, y_train)

    y_predict = mlt.predict(x_test)

    print("预测的文章类别为：", y_predict)
    # 得出准确率
    print("准确率为：", mlt.score(x_test, y_test))
    print("每个类别的精确率和召回率：", classification_report(y_test, y_predict, target_names=news.target_names))
```

4.模型调优

交叉验证和网格搜索

交叉验证：将拿到的数据，分为训练和验证集 

例：将数据分成5份，其中一份作为验证集。然后经过5次(组)的测试，每次都更换不同的验证集。即得到5组模型的结果，取均值作为最终结果。又称5折交叉验证。

超参数搜索-网格搜索：模型里需要需要手动指定的参数叫超参数 

API ：sklearn.model_selection.GridSearchCV 

sklearn.model_selection.GridSearchCV(estimator, param_grid=None,cv=None) 

​	estimator：估计器对象

​	param_grid：估计器参数(dict){“n_neighbors”:[1,3,5]}

​	cv：指定几折交叉验证

5.分类算法-决策树、随机森林 

决策树的划分依据之一-信息增益 

信息增益表示得知特征X的信息而使得类Y的信息的不确定性减少的程度 

常见决策树使用的算法 ：

•ID3     信息增益 最大的准则

•C4.5    信息增益比 最大的准则

•CART 

​	回归树: 平方误差 最小 

​	分类树: 基尼系数   最小的准则 在sklearn中可以选择划分的原则

分类决策树：class sklearn.tree.DecisionTreeClassifier(criterion=’gini’,max_depth=None,random_state=None)

​	criterion:默认是’gini’系数，也可以选择信息增益的熵’entropy’

​	max_depth:树的深度大小

​	random_state:随机数种子

优点：

​	简单的理解和解释，树木可视化。

​	需要很少的数据准备，其他技术通常需要数据归一化，

缺点：

​	决策树学习者可以创建不能很好地推广数据的过于复杂的树，这被称为过拟合。

​	决策树可能不稳定，因为数据的小变化可能会导致完全不同的树被生成

改进：减枝cart算法、随机森林

随机森林API（分类）：

​	class sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’,

​		 max_depth=None, bootstrap=True, random_state=None)

​	n_estimators：integer，optional（default = 10） 森林里的树木数量

​	criteria：string，可选（default =“gini”）分割特征的测量方法

​	max_depth：integer或None，可选（默认=无）树的最大深度 

​	bootstrap：boolean，optional（default = True）是否在构建树时使用放回抽样 

```python
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier

def decision():
    """
    决策树或随机森林对泰坦尼克号进行预测生死
    :return: None
    """
    # 获取数据
    titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

    # 处理数据，找出特征值和目标值
    x = titan[['pclass', 'age', 'sex']]
    y = titan['survived']

    print(x)
    # 缺失值处理
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 分割数据集到训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行处理（特征工程）特征-》类别-》one_hot编码
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    print(dict.get_feature_names())
    x_test = dict.transform(x_test.to_dict(orient="records"))

    # print(x_train)
    # 用决策树进行预测
    # dec = DecisionTreeClassifier()
    #
    # dec.fit(x_train, y_train)
    #
    # # 预测准确率
    # print("预测的准确率：", dec.score(x_test, y_test))
    #
    # # 导出决策树的结构
    # export_graphviz(dec, out_file="./tree.dot", feature_names=['年龄', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])

    # 随机森林进行预测 （超参数调优）
    rf = RandomForestClassifier(n_jobs=-1)
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}

    # 网格搜索与交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(x_train, y_train)

    print("准确率：", gc.score(x_test, y_test))
    print("查看选择的参数模型：", gc.best_params_)
```

6.回归算法-线性回归、岭回归

线性回归通过一个或者多个自变量与因变量之间之间进行建模的回归分析。其中特点为一个或多个称为回归系数的模型参数的线性组合 

sklearn线性回归正规方程、梯度下降API (主要针对损失函数)

正规方程:

​	sklearn.linear_model.LinearRegression()

​	普通最小二乘线性回归         coef_：回归系数

梯度下降:

​	sklearn.linear_model.SGDRegressor( )

​	通过使用SGD最小化线性模型     coef_：回归系数

小规模数据：LinearRegression(不能解决拟合问题，计算耗时)

大规模数据：SGDRegressor

回归性能评估 ：均方误差     sklearn.metrics.mean_squared_error 

mean_squared_error(y_true, y_pred)

​	y_true:真实值      y_pred:预测值

​	return:     浮点数结果

```python
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor,  Ridge
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def mylinear():
    """
    线性回归直接预测房子价格
    :return: None
    """
    # 获取数据
    lb = load_boston()

    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    print(y_train, y_test)

    # 进行标准化处理 目标值处理
    # 特征值和目标值是都必须进行标准化处理, 实例化两个标准化API
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train)
    y_test = std_y.transform(y_test)

    # 预测房价结果
    model = joblib.load("./tmp/test.pkl")
    y_predict = std_y.inverse_transform(model.predict(x_test))

    print("保存的模型预测的结果：", y_predict)

    # estimator预测
    # 正规方程求解方式预测结果
    # lr = LinearRegression()
    # lr.fit(x_train, y_train)
    # print(lr.coef_)

    # 保存训练好的模型
    # joblib.dump(lr, "./tmp/test.pkl")

    # # 预测测试集的房子价格
    # y_lr_predict = std_y.inverse_transform(lr.predict(x_test))
    # print("正规方程测试集里面每个房子的预测价格：", y_lr_predict)
    # print("正规方程的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))
    
    # 梯度下降去进行房价预测
    # sgd = SGDRegressor()
    # sgd.fit(x_train, y_train)
    # print(sgd.coef_)
    
    # 预测测试集的房子价格
    # y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
    # print("梯度下降测试集里面每个房子的预测价格：", y_sgd_predict)
    # print("梯度下降的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))
    
    # 岭回归去进行房价预测
    # rd = Ridge(alpha=1.0)
    # rd.fit(x_train, y_train)
    # print(rd.coef_)
    
    # 预测测试集的房子价格
    # y_rd_predict = std_y.inverse_transform(rd.predict(x_test))
    # print("梯度下降测试集里面每个房子的预测价格：", y_rd_predict)
    # print("梯度下降的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))
```

7.过拟合和欠拟合

过拟合：一个假设在训练数据上能够获得比其他假设更好的拟合， 但是在训练数据外的数据集上却不能很好地拟		          合数据，此时认为这个假设出现了过拟合的现象。(模型过于复杂) 

欠拟合：一个假设在训练数据上不能获得更好的拟合， 但是在训练数据外的数据集上也不能很好地拟合数据，此时认为这个假设出现了欠拟合的现象。(模型过于简单) 

过拟合原因以及解决办法 ：

​	原因：原始特征过多，存在一些嘈杂特征， 模型过于复杂是因为模型尝试去兼顾各个测试数据点

​	解决办法：

​		进行特征选择，消除关联性大的特征(很难做)

​		交叉验证(让所有数据都有过训练)

​		正则化(了解)

具有l2正则化的线性最小二乘法（岭回归） ：

​	sklearn.linear_model.Ridge(alpha=1.0) 

​	alpha:正则化力度      coef_:回归系数

岭回归：回归得到的回归系数更符合实际，更可靠。另外，能让估计参数的波动范围变小，变的更稳定。在存在病态数据偏多的研究中有较大的实用价值

8.分类算法-逻辑回归 

逻辑回归是解决二分类问题的利器 

与线性回归原理相同,但由于是分类问题，损失函数不一样，只能通过梯度下降求解

api：sklearn.linear_model.LogisticRegression(penalty=‘l2’, C=1.0) 

优点：适合需要得到一个分类概率的场景

缺点：当特征空间很大时，逻辑回归的性能不是很好（看硬件能力）

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def logistic():
    """
    逻辑回归做二分类进行癌症预测（根据细胞的属性特征）
    :return: NOne
    """
    # 构造列标签名字
    column = ['Sample code number','Clump Thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion', 'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

    # 读取数据
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=column)
    print(data)

    # 缺失值进行处理
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna()

    # 进行数据的分割
    x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]], test_size=0.25)

    # 进行标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 逻辑回归预测
    lg = LogisticRegression(C=1.0)
    lg.fit(x_train, y_train)
    print(lg.coef_)

    y_predict = lg.predict(x_test)
    print("准确率：", lg.score(x_test, y_test))
    print("召回率：", classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"]))
```

9.非监督学习 -k-means 

k-means步骤 :

​	1、随机设置K个特征空间内的点作为初始的聚类中心

​	2、对于其他每个点计算到K个中心的距离，未知的点选择最近的一个聚类中心点作为标记类别

​	3、接着对着标记的聚类中心之后，重新计算出每个聚类的新中心点（平均值）

​	4、如果计算得出的新中心点与原中心点一样，那么结束，否则重新进行第二步过程

API:  sklearn.cluster.KMeans(n_clusters=8, init=‘k-means++’, n_init=10) 

n_clusters:开始的聚类中心数量 (分类数)      init:初始化方法，默认为'k-means ++’     n_init：附近多少个点

Kmeans性能评估指标 ：轮廓系数 

api：sklearn.metrics.silhouette_score(X, labels) 

​	X：特征值        labels：被聚类标记的目标值      return：float

轮廓系数的值是介于 [-1,1] ，越趋近于1代表内聚度和分离度都相对较优 

优点：采用迭代式算法，直观易懂并且非常实用

缺点：容易收敛到局部最优解(多次聚类)

​            需要预先设定簇的数量(k-means++解决)

聚类使用：一般要做在分类之前

```python
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# 读取四张表的数据
prior = pd.read_csv("./data/instacart/order_products__prior.csv")
products = pd.read_csv("./data/instacart/products.csv")
orders = pd.read_csv("./data/instacart/orders.csv")
aisles = pd.read_csv("./data/instacart/aisles.csv")

# 合并四张表到一张表  （用户-物品类别）
_mg = pd.merge(prior, products, on=['product_id', 'product_id'])
_mg = pd.merge(_mg, orders, on=['order_id', 'order_id'])
mt = pd.merge(_mg, aisles, on=['aisle_id', 'aisle_id'])

# 交叉表（特殊的分组工具）
cross = pd.crosstab(mt['user_id'], mt['aisle'])

# 进行主成分分析
pca = PCA(n_components=0.9)  # 保留90%的特征
data = pca.fit_transform(cross)

# 把样本数量减少
x = data[:500]

# 假设用户一共分为四个类别
km = KMeans(n_clusters=4)
km.fit(x)
predict = km.predict(x)   # 聚类目标值结果array

# 显示聚类的结果
plt.figure(figsize=(10,10))

# 建立四个颜色的列表
colored = ['orange', 'green', 'blue', 'purple']
colr = [colored[i] for i in predict]
plt.scatter(x[:, 1], x[:, 20], color=colr)

plt.xlabel("1")
plt.ylabel("20")
plt.show()

# 评判聚类效果，轮廓系数
silhouette_score(x, predict)
```

10.SVM

SVM主要针对小样本数据进行学习、分类和预测（有时也叫回归）的一种方法，能解决神经网络不能解决的过学习问题，而且有很好的泛化能力

API：from sklearn.svm import SVC(分类), SVR(回归)

一般使用高斯核：kernel='rbf'

目的：SVM的目的是要找到一个线性分类的最佳超平面 f(x)=xw+b=0。求 w 和 b。

​	    首先通过两个分类的最近点，找到f(x)的约束条件。

​	    有了约束条件，就可以通过拉格朗日乘子法和KKT条件来求解，这时，问题变成了求拉格朗日乘子αi 和 b。

​	    对于异常点的情况，加入松弛变量ξ来处理。

​	    非线性分类的问题：映射到高维度、使用核函数。

解决的问题：

- 线性分类

在训练数据中，每个数据都有n个的属性和一个二类类别标志，我们可以认为这些数据在一个n维空间里。我们的目标是找到一个n-1维的超平面（hyperplane），这个超平面可以将数据分成两部分，每部分数据都属于同一个类别。 其实这样的超平面有很多，我们要找到一个最佳的。因此，增加一个约束条件：这个超平面到每边最近数据点的距离是最大的。也成为最大间隔超平面（maximum-margin hyperplane）。这个分类器也成为最大间隔分类器（maximum-margin classifier）。 支持向量机是一个二类分类器。

- 非线性分类

SVM的一个优势是支持非线性分类。它结合使用拉格朗日乘子法和KKT条件，以及核函数可以产生非线性分类器。

【关键词】支持向量，最大几何间隔，拉格朗日乘子法

```python
# 使用多种核函数对iris数据集进行分类
import sklearn.datasets as datasets
from sklearn.svm import SVC, LinearSVC

# 创建支持向量机的模型：'linear', 'poly'(多项式), 'rbf'(Radial Basis Function:基于半径函数)
svc_linear = SVC(kernel='linear')
svc_rbf = SVC(kernel='rbf')
svc_poly = SVC(kernel='poly')
linearSVC = LinearSVC()

iris = datasets.load_iris()
data = iris.data[:,[2,3]]
target = iris.target

# 训练模型
svc_linear.fit(data,target)
svc_rbf.fit(data,target)
svc_poly.fit(data,target)
linearSVC.fit(data,target)

# 预测数据，依然从图片背景中获取一系列的点
xmin,xmax = data[:,0].min()-0.5,data[:,0].max()+0.5
ymin,ymax = data[:,1].min()-0.5,data[:,1].max() +0.5
x = np.linspace(xmin,xmax,400)
y = np.linspace(ymin,ymax,200)
xx,yy = np.meshgrid(x,y)
# 点数据
xy = np.c_[xx.reshape(-1),yy.reshape(-1)]

# 预测并绘制图形for循环绘制图形
linear_y_ = svc_linear.predict(xy)
rbf_y_ = svc_rbf.predict(xy)
poly_y_ = svc_poly.predict(xy)
linearSVC_y_ = linearSVC.predict(xy)
y_ = [linear_y_,rbf_y_,poly_y_,linearSVC_y_]

plt.figure(figsize=(12,9))
titles = ['svc_linear','svc_rbf','svc_poly','linearSVC']

for i,result in enumerate(y_):
    axes = plt.subplot(2,2,i+1) 
    axes.set_title(titles[i]) 
    # 预测的结果进行展示
    axes.pcolormesh(xx,yy,result.reshape(xx.shape),cmap = 'gray')
    axes.scatter(data[:,0],data[:,1],c = target)
```







