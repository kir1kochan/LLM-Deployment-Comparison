from transformers import AutoTokenizer, AutoModel, TextStreamer
import torch

# 模型路径
model_name = "/mnt/data/chatglm3-6b"

# 测试问题
prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 加载模型
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float32  # CPU 使用 float32
).eval()

# 准备输入
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")  # 确保输入在 CPU 上

# 初始化 TextStreamer 用于流式输出
streamer = TextStreamer(tokenizer)

# 生成回答
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],  # ChatGLM3 需要 attention_mask
    max_new_tokens=300,
    streamer=streamer
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n模型回答：\n", response)