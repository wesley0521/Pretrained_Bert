from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# 下載模型

model_name = "ckiplab/bert-base-chinese"
cache_dir = "model/ckiplab/bert-base-chinese"

BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
BertForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)




model_name = r"C:\HuggingFace\model\ckiplab\bert-base-chinese\models--ckiplab--bert-base-chinese\snapshots\efe27bb4a9373384e0120ffe1cf327714ceb61bf"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device="cuda")

result = classifier("你好，我是一款語言模型")
# # 字典操作
# vocab = tokenizer.get_vocab()
# print("陽光" in vocab)
# # 添加新詞
# tokenizer.add_tokens(new_tokens = ["陽光", "大地"])
# # 添加特殊符號
# tokenizer.add_special_tokens({"eos_token": "[EOS]"})
# vocab_1 = tokenizer.get_vocab()
# print("陽光" in vocab_1)

# # 編碼
# text = "陽光照在大地上[EOS]"
# encoded = tokenizer.encode(text,
#                            text_pair = None, 
#                            truncation = True, 
#                            padding = "max_length", 
#                            max_length = 10, 
#                            add_special_tokens = True, 
#                            return_tensors = None)
# print(encoded)

