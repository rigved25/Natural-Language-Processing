import torch
from torch.utils.data import DataLoader
from torchtext import datasets

from torchtext.legacy import data
from torchtext.legacy import datasets

# Imports for plotting our result curves
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

# Visualizing POS tagged sentence
def visualizeSentenceWithTags(text,udtags):
    print("Token"+"".join([" "]*(15))+"POS Tag")
    print("---------------------------------")
    for w, t in zip(text, udtags):
        print(w+"".join([" "]*(20-len(w)))+t)

# Function to combine data elements from a batch
def pad_collate(batch):
    xx = [b[0] for b in batch]
    yy = [b[1] for b in batch]

    x_lens = [len(x) for x in xx]

    return xx, yy, x_lens

# # Create data pipeline
# train_data = datasets.UDPOS(split="train")

# # Make data loader
# train_loader = DataLoader(dataset=train_data, batch_size=500,
#                             shuffle=True, num_workers=1,
#                             drop_last=True, collate_fn=pad_collate)

# Look at the first batch
# xx,yy,xlens = next(iter(train_loader))


# visualizeSentenceWithTags(xx[0],yy[0])


TEXT = data.Field(lower = True) 
UD_TAGS = data.Field(unk_token = None)
fields = (("text", TEXT), ("udtags", UD_TAGS))
train_data , valid_data , test_data = datasets.UDPOS.splits(fields)

TEXT.build_vocab(train_data, vectors = "glove.6B.100d")
UD_TAGS.build_vocab(train_data)

print(len(train_data.examples))
# for i in range(2500, 2600):
#     print(vars(train_data.examples[i]))

print(len(valid_data.examples))
print(len(test_data.examples))

tag_count_dict = UD_TAGS.vocab.freqs.most_common()
tags = []
cnts = []
for tag, count in tag_count_dict:
    tags.append(tag)
    cnts.append(count)

plt.bar(tags, cnts)
plt.xticks(rotation=90)
plt.xlabel('Tags')
plt.ylabel('Frequency')
plt.savefig('tag_frequency_histogram.png') 

total_no_tag = sum([count for tag, count in tag_count_dict])

print("\nTag\t\tCount\t\tPercentage\n")
for tag, count in tag_count_dict:
    print(f"\t\t{tag}\t\t{count}\t\t{count/total_no_tag*100:4.2f}%")