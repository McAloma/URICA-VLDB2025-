import sys
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
import matplotlib.pyplot as plt

def plot_line_chart(data, labels=None, save_path=None):
    # 横轴
    x_values = [1, 2, 4, 8, 16, 32]

    # 检查输入数据格式
    for line in data:
        if len(line) != 6:
            raise ValueError("每个输入列表必须包含6个值。")

    # 如果没有提供标签，自动生成
    if labels is None:
        labels = [f"Line {i+1}" for i in range(len(data))]

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    for i, y_values in enumerate(data):
        plt.plot(x_values, y_values, marker='o', label=labels[i])

    # 图例、标题和轴标签
    plt.legend()
    plt.title("Time vs Scaling Factors")
    plt.xlabel("Scaling Factor")
    plt.ylabel("Time")
    plt.xscale('log', base=2)  # 横轴对数刻度
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        print(f"图像已保存到: {save_path}")

    # 显示图形
    plt.show()

# 示例数据
example_data = [
    [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],  # 第一条线
    [0.6, 1.2, 2.4, 4.8, 9.6, 19.2]   # 第二条线
]

# 示例标签
example_labels = ["Experiment A", "Experiment B"]

# 保存路径
save_path = "src/draw/pics/speed_line_chart.png"

# 调用绘图函数
plot_line_chart(example_data, example_labels, save_path)
