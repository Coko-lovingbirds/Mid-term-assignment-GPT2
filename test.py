import torch
from model import GPT, GPTConfig
from transformers import GPT2Tokenizer
from transformers import BertTokenizer
import re

# 加载分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 使用与训练时相同的配置初始化模型
config = GPTConfig(
    vocab_size=50304,
    block_size=1024,
    n_layer=3,
    n_head=6,
    n_embd=768,
    dropout=0.0,
    bias=True
)
model = GPT(config)

# 加载保存的模型权重
model_load_path = './GPT2_trained/trained_gpt_model_poetry.pth'
model.load_state_dict(torch.load(model_load_path))

# 将模型移动到 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# 文本生成函数
def generate_text(prompt, max_new_tokens=50, temperature=1.0, top_k=None):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 截断文本
        if '。' in generated_text or '，' in generated_text:
            generated_text = generated_text.split('。')[0] + '。'

        return generated_text

# 诗句生成函数
def poem(prompt):
    generated_text = generate_text(prompt, max_new_tokens=100)
    hanzi_list = re.findall(r'[\u4e00-\u9fff]', generated_text)
    hanzi_10_list  = ''.join(hanzi_list[:10])

    cycle_ = 0
    while len(hanzi_10_list) < 10 and cycle_ < 5:
        generated_text = generate_text(prompt, max_new_tokens=200)
        hanzi_list = re.findall(r'[\u4e00-\u9fff]', generated_text)
        hanzi_10_list = ''.join(hanzi_list[:10])
        cycle_ += 1

    len_s = int(len(hanzi_10_list)/2)
    sentence = hanzi_10_list[0:len_s] + "，" + hanzi_10_list[len_s:len(hanzi_10_list)] + "。"

    return sentence

if __name__ == '__main__':
    user_inputs = ["江梅", "残阳", "黄昏", "蛾眉", "灯前", "皎月", "阳光", "秋雨", "疏雨", "冷香"]
    for prompt in user_inputs:
        generated_text = poem(prompt)
        print("输入提示：", prompt)
        print("生成文本：", generated_text)
        print("\n")
