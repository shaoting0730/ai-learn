'''
Author: shaoting0730 510738319@qq.com
Date: 2026-02-24 17:26:57
LastEditors: shaoting0730 510738319@qq.com
LastEditTime: 2026-02-26 18:43:16
FilePath: /undefined/Users/zhoushaoting/Desktop/GitHub/other-learn/phthon学习/入门中/day3.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 加载数据
data = load_iris()
X = data.data
y = data.target

# 🔥 分割数据（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# 创建模型
model1 = LogisticRegression(max_iter=200)
model2 = DecisionTreeClassifier()
model3 = KNeighborsClassifier(n_neighbors=1)

# 训练模型（只用训练集）
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# 在测试集上评估
print("逻辑回归测试准确率:", model1.score(X_test, y_test))
print("决策树测试准确率:", model2.score(X_test, y_test))
print("K=1 的 KNN 测试准确率:", model3.score(X_test, y_test))