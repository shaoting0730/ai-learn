'''
Author: shaoting0730 510738319@qq.com
Date: 2026-03-02 15:32:27
LastEditors: shaoting0730 510738319@qq.com
LastEditTime: 2026-03-02 15:34:35
FilePath: /bunny/Users/zhoushaoting/Desktop/GitHub/other-learn/AI入门/入门中/机器学习Machine Learning(ML)/ai_demo.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# 1️⃣ 生成数据
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# 2️⃣ 训练SVM模型
model = SVC(kernel="rbf")
model.fit(X, y)

# 3️⃣ 画原始数据点
plt.scatter(X[:, 0], X[:, 1], c=y)

# 4️⃣ 画决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy)
Z = Z.reshape(XX.shape)

plt.contour(XX, YY, Z, levels=[0])
plt.title("SVM 自动找到分界线")
plt.show()