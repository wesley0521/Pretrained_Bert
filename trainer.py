import torch
from torch.utils.data import DataLoader
from mydata import MyDatasets
from net import Model
from transformers import BertTokenizer
from torch.optim import AdamW
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 設置超參數
epochs = 100
token = BertTokenizer.from_pretrained(r"C:\HuggingFace\model\ckiplab\bert-base-chinese\models--ckiplab--bert-base-chinese\snapshots\efe27bb4a9373384e0120ffe1cf327714ceb61bf")
# 對數據編碼
def collate_fn(data):
    sentences = [i[0] for i in data]
    labels = [i[1] for i in data]
    # 對數據進行編碼
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sentences,
        padding = "max_length",
        max_length = 350,
        truncation = True,
        return_tensors = "pt",
        return_length = True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(labels)
    return input_ids,attention_mask,token_type_ids,labels

# 設置數據
train_dataset = MyDatasets("train")
# 設置 DataLoader
train_loader = DataLoader(train_dataset,
                          batch_size=32,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=collate_fn)

if __name__ == "__main__":
    # 設置模型
    print(f"使用 {DEVICE} 設備")
    model = Model().to(DEVICE)
    optimizer = AdamW(model.parameters(),lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for i,(input_ids,attention_mask,token_type_ids,labels) in enumerate(train_loader):
            # 將數據轉換為 GPU 設備
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            # 前向傳播
            out = model(input_ids,attention_mask,token_type_ids)
            # 計算損失
            loss = loss_fn(out,labels)
            # 清空權重
            optimizer.zero_grad()
            # 反向傳播
            loss.backward()
            # 更新權重
            optimizer.step()

            if i % 5 == 0:
                out = out.argmax(dim=1)     # 沿著每個樣本計算最大值的索引
                acc = (out == labels).sum().item() / len(labels)    # .item() 將張量轉換為標量(int)
                print(f"epoch:{epoch},step:{i},loss:{loss.item()},acc:{acc}")

    torch.save(model.state_dict(),f"params/{epoch}bert.pt")
    print(f"epoch:{epoch} 訓練完成")
