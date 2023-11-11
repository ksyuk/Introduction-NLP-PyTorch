# %%
import torch

from tensor import train_dataset, test_dataset, encode, vocab
vocab_size = len(vocab)

# %%
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else device)

# %%
# 文章をBoWに変換
# 文章中に出現する単語毎の出現回数をカウント
def to_bow(text, bow_vocab_size=len(vocab)):
    tensor = torch.zeros(bow_vocab_size, dtype=torch.float32)
    for i in encode(text):
        if i < bow_vocab_size:
            tensor[i] += 1
    return tensor

# to_bow(first_sentence)
# tensor([2., 1., 2.,  ..., 0., 0., 0.])

# %%
import time

from torch.utils.data import DataLoader

# this collate function gets list of batch_size tuples, and needs to 
# return a pair of label-feature tensors for the whole minibatch
# データセット中の全ての文章をBoWに変
def bowify(batch):
    labels = []
    features = []

    start = time.time()
    print("bowify start")
    for label, feature in batch:
        labels.append(label-1)
        features.append(to_bow(feature))
    end = time.time()
    print("time: ", end - start)
    print("bowify end")

    return (
        torch.LongTensor(labels),
        torch.stack(features)
    )

# データセットをBoWに変換
train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=bowify, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=bowify, shuffle=True)

# %%
# BoWを基にしたclassifierニューラルネットワーク
# linear層だけを持ったニューラルネットワークを作成
# 活性化関数にlogsoftmaxを使用
network = torch.nn.Sequential(torch.nn.Linear(vocab_size, 4), torch.nn.LogSoftmax(dim=1))

# %%
def train_epoch(network, dataloader, learning_rate=0.01, optimizer=None, loss_fn=torch.nn.NLLLoss(), epoch_size=None, report_freq=200):
    print("traing...")

    optimizer = optimizer or torch.optim.Adam(network.parameters(), lr=learning_rate)
    # ネットワークにトレーニングすると伝える
    network.train()

    total_loss, accuracy, count, i = 0, 0, 0, 0
    for labels, features in dataloader:
        print("batch start")
        # labels: bowify返り値の0番目の要素
        # features: bowify返り値の1番目の要素
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        out = network(features)
        loss = loss_fn(out, labels) #cross_entropy(out,labels)

        loss.backward()
        optimizer.step()
        total_loss += loss
        _, predicted = torch.max(out, 1)
        accuracy += (predicted==labels).sum()
        count += len(labels)

        i += 1
        if i % report_freq == 0:
            print(f"{count}: accuracy={accuracy.item()/count}")

        if epoch_size and count > epoch_size:
            print(epoch_size)
            print(count)
            break

    return total_loss.item()/count, accuracy.item()/count

# %%
# train_epoch(network, train_loader, epoch_size=2)

# %%
N = 10
def count_df():
    df = torch.zeros(vocab_size)
    for _, line in train_dataset[:N]:
        for i in set(encode(line)):
            df[i] += 1
    return df

def crate_tf_idf(s):
    bow = to_bow(s)
    return bow * torch.log((N+1)/(count_df()+1))

# crate_tf_idf(train_dataset[0][1])
# tensor([2.5986, 1.2993, 0.0000,  ..., 0.0000, 0.0000, 0.0000])


