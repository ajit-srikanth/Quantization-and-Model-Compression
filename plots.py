import matplotlib.pyplot as plt
import numpy as np

# Data for the graphs
metrics = ["Model Size", "Perplexity", "Inference Speed"]
# before_quant = [1324.78, 19.1325, 3.61]
# after_quant = [416.56, 19.1471, 3.38]
# partial_quant = [1277.65, 19.1338, 3.57]


# # model_sizes = [1324.785664, 359.354368, 207.835136, 207.835136]  # Default, 8-bit, 4-bit, nf4
# # perplexities = [19.1325, 19.1612, 22.8055, 21.0444]  # Default, 8-bit, 4-bit, nf4
# # inference_speeds = [3.60, 5.75, 7.26, 7.21]  # Default, 8-bit, 4-bit, nf4
before_quant = [1324.78, 19.13, 3.60]  # Model size, Perplexity, Inference speed for Default
eightbit_quant = [359.35, 19.16, 5.75]  # Model size, Perplexity, Inference speed for 8-bit
fourbit_quant = [207.83, 22.80, 7.26]  # Model size, Perplexity, Inference speed for 4-bit
nf4_quant = [207.83, 21.04, 7.21]  # Model size, Perplexity, Inference speed for NF4

x = np.arange(len(metrics))  # the label locations
width = 0.2  # the width of the bars

fig, ax1 = plt.subplots()

# Plotting Model Size
rects1 = ax1.bar(x - 1.5*width, [before_quant[0], 0, 0], width, label='Before Quantization', color='blue')
rects2 = ax1.bar(x - 0.5*width, [eightbit_quant[0], 0, 0], width, label='8-bit Quantization', color='orange')
rects3 = ax1.bar(x + 0.5*width, [fourbit_quant[0], 0, 0], width, label='4-bit Quantization', color='green')
rects4 = ax1.bar(x + 1.5*width, [nf4_quant[0], 0, 0], width, label='NF4 Quantization', color='red')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_xlabel('Metrics')
ax1.set_ylabel('Model Size')
ax1.set_title('Comparison of Model Metrics Before and After Quantization')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend(loc='upper left')

# Creating another y-axis for Perplexity
ax2 = ax1.twinx()
rects5 = ax2.bar(x - 1.5*width, [0, before_quant[1], 0], width, label='Before Quantization', color='blue')
rects6 = ax2.bar(x - 0.5*width, [0, eightbit_quant[1], 0], width, label='8-bit Quantization', color='orange')
rects7 = ax2.bar(x + 0.5*width, [0, fourbit_quant[1], 0], width, label='4-bit Quantization', color='green')
rects8 = ax2.bar(x + 1.5*width, [0, nf4_quant[1], 0], width, label='NF4 Quantization', color='red')

ax2.set_ylabel('Perplexity')
ax2.set_ylim(19, 23)

# Creating another y-axis for Inference Speed
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
rects9 = ax3.bar(x - 1.5*width, [0, 0, before_quant[2]], width, label='Before Quantization', color='blue')
rects10 = ax3.bar(x - 0.5*width, [0, 0, eightbit_quant[2]], width, label='8-bit Quantization', color='orange')
rects11 = ax3.bar(x + 0.5*width, [0, 0, fourbit_quant[2]], width, label='4-bit Quantization', color='green')
rects12 = ax3.bar(x + 1.5*width, [0, 0, nf4_quant[2]], width, label='NF4 Quantization', color='red')

ax3.set_ylabel('Inference Speed')

# Function to add labels on top of the bars
def add_labels(rects, ax):
    for rect in rects:
        height = rect.get_height()
        if height == 0:
            continue
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1, ax1)
add_labels(rects2, ax1)
add_labels(rects3, ax1)
add_labels(rects4, ax1)
add_labels(rects5, ax2)
add_labels(rects6, ax2)
add_labels(rects7, ax2)
add_labels(rects8, ax2)
add_labels(rects9, ax3)
add_labels(rects10, ax3)
add_labels(rects11, ax3)
add_labels(rects12, ax3)

fig.tight_layout()

plt.show()
