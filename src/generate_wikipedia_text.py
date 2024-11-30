import os
from datasets import load_dataset

dataset = load_dataset("wikipedia", "20220301.en", split='train', streaming=True)
shuffle_dataset = dataset.shuffle(seed=70) # type: ignore

wikipedia_samples = []

# load 1000 sample articles from shuffled dataset
shuffle_iter = iter(shuffle_dataset)
for _ in range (100):
    wikipedia_samples.append(next(shuffle_iter))
print(f"loaded {len(wikipedia_samples)} wikipedia articles")
 # clear wikipedia_extracted folder
if not os.path.exists("./test_docs/wikipedia_extracted/"):
    os.mkdir("./test_docs/wikipedia_extracted/")
else:
    for file in os.listdir("./test_docs/wikipedia_extracted/"):
        os.remove(f"./test_docs/wikipedia_extracted/{file}")
for sample in wikipedia_samples:
   
    
    with open(f"test_docs/wikipedia_extracted/{sample['title']}.txt", "w") as f:
        f.write(sample["title"] + "\n")
        f.write(sample["text"])
        f.write("\n"+f"Source: {sample['url']}")
    
print(f"done generating {len(wikipedia_samples)} wikipedia articles")