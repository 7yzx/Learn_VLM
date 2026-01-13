import numpy as np

# --- 关键步骤 1: 设置后端 (必须在 import pyplot 之前) ---
import matplotlib
matplotlib.use('Agg')  # 'Agg' 表示不使用图形界面，专门用于生成图像文件
import matplotlib.pyplot as plt

def sinusoidal_position_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis] # [seq_len, 1]
    freq = np.power(10000, (2 * np.arange(d_model//2)/d_model)) # [d_model/2 , ]
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(pos/freq)
    pe[:, 1::2] = np.cos(pos/freq)
    
    return pe

# 参数设置
seq_len = 512  # 序列长度
d_model = 128  # 模型的维度

# 获取位置编码
pe = sinusoidal_position_encoding(seq_len, d_model)

# 绘图
plt.figure(figsize=(10, 6))
fixed_pos = 10 
for i in [0, 32, 64]: 
    plt.plot(np.arange(seq_len), pe[:, i], label=f'Dimension i={i}')

plt.xlabel("Position (pos)")
plt.ylabel("Position Encoding Value")
plt.title(f"Position Encoding (Sinusoidal)")
plt.legend(loc='upper right')
plt.tight_layout()

# --- 关键步骤 2: 使用 savefig 保存 ---
save_path = "position_vis.png"
plt.savefig(save_path, dpi=300) # dpi=300 保证图片高清
print(f"图片已成功保存至: {save_path}")

# 注意：服务器上绝对不要写 plt.show()，否则会报错