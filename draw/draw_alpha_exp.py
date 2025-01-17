import matplotlib.pyplot as plt
import numpy as np



x_labels = ["224", "200", "150", "100", "80", "60", "50"] 
x_positions = np.arange(len(x_labels))  

similarity_lines = {
    "Sim@1": [0.8775, 0.9322, 0.8531, 0.9143, 0.9528, 0.9473, 0.9185],
    "Sim@3": [0.8652, 0.9237, 0.8434, 0.9067, 0.9440, 0.9404, 0.9125],
    "Sim@5": [0.8554, 0.9068, 0.8298, 0.9011, 0.9384, 0.9343, 0.9081],
}
iou_lines = {
    "IoU@1": [0.5306, 0.5692, 0.5877, 0.6223, 0.6648, 0.6516, 0.6427],
    "IoU@3": [0.4544, 0.5132, 0.5216, 0.5641, 0.6141, 0.6003, 0.6030],
    "IoU@5": [0.3782, 0.4395, 0.4512, 0.5251, 0.5649, 0.5518, 0.5648],
}
runtime = [46.19, 47.45, 49.83, 60.41, 68.94, 93.04, 92.34]

fig, ax1 = plt.subplots(figsize=(10, 6))

for label, data in similarity_lines.items():
    ax1.plot(x_positions, data, marker='o', label=f"Similarity - {label}")

ax1.set_xlabel("Step of Tessellation of query image and database.", fontsize=12)
ax1.set_ylabel("Similarity of Retrieval Results.", fontsize=12, color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
for label, data in iou_lines.items():
    ax2.plot(x_positions, data, marker='s', linestyle='--', label=f"IoU - {label}")

ax2.set_ylabel("IoU Value of Retrieval Results.", fontsize=12, color="green")
ax2.tick_params(axis="y", labelcolor="green")

ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60)) 
bar_width = 0.4 
bars = ax3.bar(
    x_positions, 
    runtime, 
    width=bar_width, 
    alpha=0.4, 
    color="gray", 
    label="Running Time"
)

for bar, value in zip(bars, runtime):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(value),
             ha="center", va="bottom", fontsize=10, color="black")

ax3.set_ylabel("Average retrieval time(s)", fontsize=12, color="gray")
ax3.tick_params(axis="y", labelcolor="gray")

ax2.legend(loc="upper right")
ax3.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3)

plt.title("Performance Metrics and Runtime with 20 WSIs in TCGA", fontsize=14)
ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

ax1.set_ylim(0.7, 1.0)
ax2.set_ylim(0.3, 1.0)
ax3.set_ylim(40, 120)

ax1.set_xticks(x_positions) 
ax1.set_xticklabels(x_labels, fontsize=12)  

plt.tight_layout()
plt.savefig("draw/pics/sorted_alpha_exp_chart.png")
plt.show()