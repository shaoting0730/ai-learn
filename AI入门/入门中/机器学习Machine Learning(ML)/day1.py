'''
Author: shaoting0730 510738319@qq.com
Date: 2026-02-24 17:26:57
LastEditors: shaoting0730 510738319@qq.com
LastEditTime: 2026-02-24 17:50:33
FilePath: /bunny/Users/zhoushaoting/Desktop/GitHub/other-learn/phthon学习/day1/day1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# 1️⃣ 加载数据
data = load_iris()
X = data.data
y = data.target

# 2️⃣ 创建模型
model = LogisticRegression(max_iter=200)

# 3️⃣ 训练模型
model.fit(X, y)

# 4️⃣ 查看准确率
score = model.score(X, y)
print("模型准确率:", score)

# 5️⃣ 预测一条数据
prediction = model.predict([X[0]])
print("真实结果:", y[0])
print("预测结果:", prediction[0])