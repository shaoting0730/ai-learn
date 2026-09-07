'''
Author: shaoting0730 510738319@qq.com
Date: 2026-02-24 17:26:57
LastEditors: shaoting0730 510738319@qq.com
LastEditTime: 2026-02-25 17:52:39
FilePath: /bunny/Users/zhoushaoting/Desktop/GitHub/other-learn/phthon学习/入门中/day2.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# 加载数据
data = load_iris()
X = data.data
y = data.target

# 创建三个不同模型
model1 = LogisticRegression(max_iter=200)
model2 = DecisionTreeClassifier()
model3 = KNeighborsClassifier(n_neighbors=10)

# 训练模型
model1.fit(X, y)
model2.fit(X, y)
model3.fit(X, y)

# 输出准确率
print("逻辑回归准确率:", model1.score(X, y))
print("决策树准确率:", model2.score(X, y))
print("KNN准确率:", model3.score(X, y))