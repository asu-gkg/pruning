import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# 创建一个图形和子图
fig, ax = plt.subplots()

# 初始化数据
data = np.random.randn(100)
n_bins = 50  # 直方图的bin数

# 初始化直方图
hist, bins = np.histogram(data, bins=n_bins)
patches = ax.bar(bins[:-1], hist, width=(bins[1] - bins[0]), edgecolor='black')

# 更新函数
def update(new_data):
    ax.cla()  # 清除轴
    hist, bins = np.histogram(new_data, bins=n_bins)
    ax.bar(bins[:-1], hist, width=(bins[1] - bins[0]), edgecolor='black')
    ax.set_xlim(-5, 5)  # 设置x轴的范围
    ax.set_ylim(0, max(hist) + 5)  # 设置y轴的范围
    ax.set_title('Dynamic Histogram of Vectors')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

# 模拟数据更新
def data_generator():
    while True:
        yield np.random.randn(100)  # 生成一个新的向量

# 创建动画
ani = FuncAnimation(fig, update, frames=data_generator, interval=1000)

plt.show()