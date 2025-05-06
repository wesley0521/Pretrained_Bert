from torch.utils.data import Dataset
from datasets import load_from_disk

class MyDatasets(Dataset):
    # 初始化
    def __init__(self,split):
        self.dataset = load_from_disk(r"C:\HuggingFace\ChnSentiCorp")
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        else:
            print("error")
    # 獲取數據集長度
    def __len__(self):
        return len(self.dataset)
    #訂製化數據
    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]
        return text,label
if __name__ == "__main__":
    dataset = MyDatasets("validation")
    for data in dataset:
        print(data)

