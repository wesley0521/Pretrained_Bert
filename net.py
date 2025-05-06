from transformers import BertModel
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 預訓練模型
pretrained_model = BertModel.from_pretrained(r"C:\HuggingFace\model\ckiplab\bert-base-chinese\models--ckiplab--bert-base-chinese\snapshots\efe27bb4a9373384e0120ffe1cf327714ceb61bf").to(DEVICE)
print(pretrained_model)

# 將預訓練模型所提取得特徵進行分類
class Model(torch.nn.Module):
    # 設計模型
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768,2)   # 全連接層
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 不計算梯度
        with torch.no_grad():
            out = pretrained_model(input_ids=input_ids,               # 把文字轉成的 token ID 序列
                                   attention_mask=attention_mask,     # 標示哪些是實際文字、哪些是 padding
                                   token_type_ids=token_type_ids)     # 用來標記兩句話的邊界
        # 將特徵進行分類
        out = self.fc(out.last_hidden_state[:,0])   # 取 cls token 的特徵 (cls token 是 BERT 用來表示句子的特徵同時也是維度為(batch_size,768)的向量)
        out = torch.nn.functional.softmax(out,dim=1)  # dim=1 表示對每個樣本進行 softmax 操作
        return out
