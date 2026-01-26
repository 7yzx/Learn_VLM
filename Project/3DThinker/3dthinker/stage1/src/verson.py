import transformers
import os

print("Transformers 版本:", transformers.__version__)
print("安装路径:", os.path.dirname(transformers.__file__))

# 检查是否存在 Qwen2.5 的文件夹
qwen_path = os.path.join(os.path.dirname(transformers.__file__), "models", "qwen2_5_vl")
if os.path.exists(qwen_path):
    print(f"✅ Qwen2.5-VL 文件夹存在于: {qwen_path}")
else:
    print(f"❌ Qwen2.5-VL 文件夹不存在！")