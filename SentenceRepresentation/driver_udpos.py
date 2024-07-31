
# Basic python imports for logging and sequence generation
import itertools
import random
import numpy as np
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Imports foe loading the UDPOS dataset
from torch.utils.data import DataLoader
# from torchtext import data
# from torchtext import datasets
from torchtext.legacy import data
from torchtext.legacy import datasets
# from torch.utils.data.backward_compatibility import worker_init_fn

# Imports for Pytorch for the things we need
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn as nn
import torch.nn.functional as F

# Imports for progress bar
from tqdm import tqdm

# Imports for plotting our result curves
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


# Set random seed for python and torch to enable reproducibility (at least on the same hardware)
random.seed(42)
torch.manual_seed(42)

# Determine if a GPU is available for use, define as global variable
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
 
# Main Driver Loop
def main():

    # Create data pipeline
    TEXT = data.Field(lower = True) 
    UD_TAGS = data.Field(unk_token = None)
    fields = (("text", TEXT), ("udtags", UD_TAGS))
    train_data , valid_data , test_data = datasets.UDPOS.splits(fields)

    # build vocab
    TEXT.build_vocab(train_data, vectors = "glove.6B.100d")
    UD_TAGS.build_vocab(train_data)

    # parameters
    input_dim = len(TEXT.vocab)
    output_dim = len(UD_TAGS.vocab)
    embedding_dim = 100
    num_layers = 1
    hidden_dim = 64
    batch_size = 512
    text_pad_token = TEXT.vocab.stoi[TEXT.pad_token]
    tag_pad_token = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]

    # Build model and put it on the GPU
    logging.info("Building model")
    model = BiDirectLSTM(input_dim, output_dim, hidden_dim, embedding_dim, padding_idx=text_pad_token, num_layers=num_layers)
    model.to(dev) # move to GPU if cuda is enabled

    # get embeddings
    glove_pretrained_embed = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(glove_pretrained_embed)
    model.embedding.weight.data[text_pad_token] = torch.zeros(embedding_dim)

    logging.info("Training model")

    # Make data loader
    train_dataloader, valid_dataloader, test_dataloader = data.BucketIterator.splits(
                                                    (train_data, valid_data, test_data), 
                                                    batch_size = batch_size,
                                                    device = dev)
    
    logging.info("Running generalization experiment")
    train_validate_test(model, train_dataloader, valid_dataloader, test_dataloader, tag_pad_token)

    sentences = [["the", "old", "man", "the", "boat", "."], 
                ["The", "complex", "houses", "married", "and", "single", "soldiers", "and", "their", "families", "."],
                ["The", "man", "who", "hunts", "ducks", "out", "on", "weekends", "."]]
    
    for sentence in sentences:
        tag_sentence(model, sentence, TEXT, UD_TAGS)


class BiDirectLSTM(torch.nn.Module) :

    def __init__(self, input_dim, output_dim, hidden_dim, embedding_dim, padding_idx, num_layers=1) :
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = padding_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)   

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (ht, ct) = self.lstm(x)
        out = self.linear(lstm_out)
        return out

def train_validate_test(model, train_loader, validate_loader, test_loader, tag_pad_token, epochs=20, lr=0.001):

    train_losses = []
    val_losses = []

    # Define a cross entropy loss function
    crit = torch.nn.CrossEntropyLoss()

    # Collect all the learnable parameters in our model and pass them to an optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    # Adam is a version of SGD with dynamic learning rates 
    # (tends to speed convergence but often worse than a well tuned SGD schedule)
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=0.0001)

    lowest_val_loss = float('Inf')

    # Main training loop over the number of epochs
    for i in tqdm(range(epochs), desc="Training: "):
        trn_loss, trn_acc = train_model(model, train_loader, crit, optimizer, tag_pad_token)
        val_loss, val_acc = validate_model(model, validate_loader, crit, tag_pad_token)

        train_losses.append(trn_loss)
        val_losses.append(val_loss)

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            torch.save(model.state_dict(), 'model.pt')

        if i % 10 == 0:
            logging.info("epoch %d train loss %.3f, train acc %.3f val loss %.3f, val acc %.3f" % (i, trn_loss, trn_acc*100, val_loss, val_acc*100))
    
    # test data
    model.load_state_dict(torch.load('model_150.pt'))
    test_loss, test_acc = validate_model(model, test_loader, crit, tag_pad_token)
    print("test loss %.3f, test acc %.3f%%" % (test_loss, test_acc*100))

    plt.title("Training, Validation loss over the course of training.")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(["training", "validation"], loc ="upper right")
    plt.grid(linestyle='solid')
    plt.savefig('training_validation_loss.png')


def train_model(model, train_loader, crit, optimizer, tag_pad_token):
    # Set model to train mode so things like dropout behave correctly
    model.train()
    sum_loss = 0.0
    sum_accuracy = 0.0
    total = 0
    correct = 0

    # for each batch in the dataset
    for b in train_loader:

        # push them to the GPU if we are using one
        text = b.text.to(dev)
        tags = b.udtags.to(dev)
        tags = tags.view(-1)

        pred_tag = model(text)
        pred_tag = pred_tag.view(-1, pred_tag.shape[-1])

        # compute the loss with respect to the true labels
        loss = crit(pred_tag, tags)
        
        # zero out the gradients, perform the backward pass, and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute loss and accuracy to report epoch level statitics
        non_pads = torch.nonzero(tags != tag_pad_token)

        max_preds = pred_tag.argmax(dim=1, keepdim=True)
        correct = max_preds[non_pads].squeeze(1).eq(tags[non_pads])
        sum_corr = correct.sum()
        total = torch.FloatTensor([tags[non_pads].shape[0]]).to(dev)

        sum_loss += loss.item()
        sum_accuracy += sum_corr/total

    return sum_loss/len(train_loader), sum_accuracy/len(train_loader)
        

     
def validate_model(model, loader, crit, tag_pad_token):
    # Set model to train mode so things like dropout behave correctly
    model.train()
    sum_loss = 0.0
    sum_accuracy = 0.0
    total = 0
    correct = 0

    # for each batch in the dataset
    for b in loader:

        # push them to the GPU if we are using one
        text = b.text.to(dev)
        tags = b.udtags.to(dev)
        tags = tags.view(-1)

        pred_tag = model(text)
        pred_tag = pred_tag.view(-1, pred_tag.shape[-1])

        # compute the loss with respect to the true labels
        loss = crit(pred_tag, tags)

        # compute loss and accuracy to report epoch level statitics
        non_pads = torch.nonzero(tags != tag_pad_token)

        max_preds = pred_tag.argmax(dim=1, keepdim=True)
        correct = max_preds[non_pads].squeeze(1).eq(tags[non_pads])
        sum_corr = correct.sum()
        total = torch.FloatTensor([tags[non_pads].shape[0]]).to(dev)

        sum_loss += loss.item()
        sum_accuracy += sum_corr/total

    return sum_loss/len(loader), sum_accuracy/len(loader)

def tag_sentence(model, sentence, TEXT, UD_TAGS):

    model.eval()
    embeddings = [TEXT.vocab.stoi[t] for t in sentence]
    preds = model(torch.LongTensor(embeddings).unsqueeze(-1).to(dev)) #added batch dim
    maxpreds = preds.argmax(-1)
    
    predicted_tags = [UD_TAGS.vocab.itos[t.item()] for t in maxpreds]
    print("-----------------------------")
    print("TAG\t TOKEN\n")
    for token, tag in zip(sentence, predicted_tags):
        print(tag, "\t", token)


if __name__== "__main__":
    main()
