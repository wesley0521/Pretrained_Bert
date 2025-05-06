from datasets import load_dataset, load_from_disk

# 在線下載數據
dataset = load_dataset(path="lansinuote/ChnSentiCorp")

print(dataset)

# 將數據保存到本地
dataset.save_to_disk("ChnSentiCorp")

# 從本地加載數據
dataset = load_from_disk("ChnSentiCorp")

for i in dataset["test"]:
    print(i)