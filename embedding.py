# %%
import torch

from tensor import train_dataset, encode, vocab, classes

vocab_size = len(vocab)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
def padify(batch):
    # batch is the list of tuples of length batch_size
    #   - first element of a tuple = label, 
    #   - second = feature (text sequence)

    # build vectorized sequence
    vectors = [encode(x[1]) for x in batch]

    # first, compute max length of a sequence in this minibatch
    l = max(map(len, vectors))

    labels = []
    for t in batch:
        labels.append(t[0]-1)
    features = []
    for t in vectors:
        features.append(torch.nn.functional.pad(torch.tensor(t), (0, l-len(t)), mode='constant', value=0))
    return ( # tuple of two tensors - labels and features
        torch.LongTensor(labels),
        torch.stack(features)
    )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=padify, shuffle=True)

# %%
class EmbedClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.fc = torch.nn.Linear(embed_dim, num_class)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        return self.fc(x)

network = EmbedClassifier(vocab_size, 32 ,len(classes)).to(device)

# %%
# train_epoch(network, train_loader, learning_rate=1, epoch_size=25)

# %%
class EmbedClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc = torch.nn.Linear(embed_dim, num_class)

    def forward(self, text, off):
        x = self.embedding(text, off)
        return self.fc(x)

# %%
def offsetify(batch):
    # first, compute data tensor from all sequences
    x = [torch.tensor(encode(t[1])) for t in batch]

    # now, compute the offsets by accumulating the tensor of sequence lengths
    o = [0] + [len(t) for t in x]
    o = torch.tensor(o[:-1]).cumsum(dim=0)

    labels = []
    for t in batch:
        labels.append(t[0]-1)

    return ( 
        torch.LongTensor(labels), # labels
        torch.cat(x), # text 
        o
    )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=offsetify, shuffle=True)

# %%
network = EmbedClassifier(vocab_size, 32, len(classes)).to(device)

def train_epoch_emb(network, dataloader, lr=0.01, optimizer=None, loss_fn=torch.nn.CrossEntropyLoss(), epoch_size=None, report_freq=200):
    print("traing...")

    optimizer = optimizer or torch.optim.Adam(network.parameters(), lr=lr)
    loss_fn = loss_fn.to(device)
    network.train()

    total_loss, accurancy, count, i = 0, 0, 0, 0
    for labels, text, off in dataloader:
        optimizer.zero_grad()
        labels, text, off = labels.to(device), text.to(device), off.to(device)
        out = network(text, off)
        loss = loss_fn(out, labels) #cross_entropy(out,labels)

        loss.backward()
        optimizer.step()
        total_loss+=loss
        _, predicted = torch.max(out,1)
        acuurancy += (predicted==labels).sum()
        count += len(labels)

        i += 1
        if i % report_freq == 0:
            print(f"{count}: acuurancy={accurancy.item()/count}")

        if epoch_size and count > epoch_size:
            break

    return total_loss.item()/count, accurancy.item()/count

# %%
# train_epoch_emb(network, train_loader, lr=4, epoch_size=25)
