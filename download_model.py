import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-zh"
save_path = "./translation_model"

print(f"正在下载模型: {model_name}")
print(f"保存路径: {os.path.abspath(save_path)}")
print("=" * 60)

try:
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    os.makedirs(save_path, exist_ok=True)
    
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    
    print("\n✅ 模型下载成功！")
    print(f"模型已保存到: {os.path.abspath(save_path)}")
    print("\n现在可以运行翻译应用了。")
    
except Exception as e:
    print(f"\n❌ 下载失败: {str(e)}")
    print("\n尝试使用代理或手动下载模型:")
    print("1. 访问 https://hf-mirror.com/Helsinki-NLP/opus-mt-en-zh")
    print("2. 下载所有文件到 ./translation_model 目录")
    print("3. 运行 streamlit run translation_app.py")
