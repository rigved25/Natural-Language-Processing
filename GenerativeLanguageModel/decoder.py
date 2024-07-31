######################################################
# Use these package versions
#!pip install torchtext==0.6.0 torch==1.13.1
######################################################


import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] =':16:8' #This is a command to reduce non-deterministic behavior in CUDA
import warnings
warnings.simplefilter("ignore", UserWarning)
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import get_tokenizer
# from torchtext.legacy.data import Field
# from torchtext.legacy import data
# from torchtext.legacy import datasets
import sys
import argparse
from LanguageModel import LanguageModel
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    chkpt = "got_language_model"

    logging.info('Using device: {}'.format(dev))

    logging.info("Loading tokenizer and vocab from vocab.pkl")  
    text_field = pickle.load(open("vocab.pkl", "rb"))
    vocab_size = len(text_field.vocab.itos)

    logging.info("Loading checkpoint {}".format(chkpt))
    lm = LanguageModel(vocab_size).to(dev)
    lm.load_state_dict(torch.load(chkpt))
    lm.eval()


    p = "the night is dark and full of terrors"

    # Torch is a bit frustrating at times and some things that ought to be deterministic are not.
    # This is an attempt to resolve that, but it doesn't work 100% of the time
    torch.use_deterministic_algorithms(True)
    seed = 42
    mlen = 150

    torch.manual_seed(seed); np.random.seed(seed)
    print("\n----------- Vanilla Sampling -----------")
    print(sample(lm, text_field, prompt=p, max_len=mlen))

    torch.manual_seed(seed); np.random.seed(seed)
    print("\n------- Temp-Scaled Sampling 0.0001 -------")
    print(sample(lm, text_field, prompt=p, temp=0.0001, max_len=mlen))

    torch.manual_seed(seed); np.random.seed(seed)
    print("\n------- Temp-Scaled Sampling 100 --------")
    print(sample(lm, text_field, prompt=p, temp=100, max_len=mlen))

    torch.manual_seed(seed); np.random.seed(seed)
    print("\n----------- Top-k Sampling 1 -----------")
    print(sample(lm, text_field, prompt=p, k=1, max_len=mlen))

    torch.manual_seed(seed); np.random.seed(seed)
    print("\n----------- Top-k Sampling 20 -----------")
    print(sample(lm, text_field, prompt=p, k=20, max_len=mlen))

    torch.manual_seed(seed); np.random.seed(seed)
    print("\n----------- Top-p Sampling 0.001 -----------")
    print(sample(lm, text_field, prompt=p, p=0.001, max_len=mlen))

    torch.manual_seed(seed); np.random.seed(seed)
    print("\n----------- Top-p Sampling 0.75 -----------")
    print(sample(lm, text_field, prompt=p, p=0.75, max_len=mlen))

    torch.manual_seed(seed); np.random.seed(seed)
    print("\n----------- Top-p Sampling 1 -----------")
    print(sample(lm, text_field, prompt=p, p=1, max_len=mlen))


    torch.manual_seed(seed); np.random.seed(seed)
    print("\n----------- Beam Search B=1 -----------")
    print(beamsearch(lm, text_field, prompt=p, beams=1, max_len=mlen))

    torch.manual_seed(seed); np.random.seed(seed)
    print("\n----------- Beam Search B=10 -----------")
    print(beamsearch(lm, text_field, prompt=p, beams=10, max_len=mlen))

    torch.manual_seed(seed); np.random.seed(seed)
    print("\n----------- Beam Search B=50 -----------")
    print(beamsearch(lm, text_field, prompt=p, beams=50, max_len=mlen))

    print()

############################################################################################
# TASK 2.1
############################################################################################

def beamsearch(model, text_field, beams=5, prompt="", max_len=50):
    
    output = text_field.process([text_field.tokenize(prompt.lower())]).to(dev)
    hidden_size = model.hidden_size
    num_layers = model.num_layers
    hidden_state = torch.zeros(num_layers, 1, hidden_size).to(dev)
    cell_state = torch.zeros(num_layers, 1, hidden_size).to(dev)
    
    crit = nn.LogSoftmax(dim=1)

    output, hidden_state, cell_state = model(output, hidden_state, cell_state)
    probs = crit(output[-1])

    hidden_state = torch.cat([hidden_state]*beams, 1)
    cell_state = torch.cat([cell_state]*beams, 1)
    
    topk_p, topk_idx = torch.topk(probs, beams)

    decodedStrings = topk_idx.view(beams, 1)

    last_prop = []
    for i in range(max_len-1):
        output, hidden_state, cell_state = model(topk_idx.to(dev), hidden_state.to(dev), cell_state.to(dev))
        probs = crit(output[-1])

        cum_log_p = probs + topk_p.view((beams, 1))

        topk_p, topk_idx = torch.topk(cum_log_p.view(-1), beams)
        beam_index = np.array(np.unravel_index(topk_idx.cpu().numpy(), cum_log_p.shape)).T

        new_ht = []
        new_ct = []
        for l in range(num_layers):
            ht_layer = []
            ct_layer = []
            for r, c in beam_index:
                ht_layer.append(hidden_state[l][r])
                ct_layer.append(cell_state[l][r])
            new_ht.append(torch.stack(ht_layer).to(dev))
            new_ct.append(torch.stack(ct_layer).to(dev))

        hidden_state = torch.stack(new_ht)
        cell_state = torch.stack(new_ct)

        strings = []
        for i, (r, c) in enumerate(beam_index):
            topk_idx[i] = c
            strings.append(torch.cat([decodedStrings[r], torch.tensor([c]).to(dev)]))

        decodedStrings = strings
        topk_idx = topk_idx.unsqueeze(0).to(dev)
        last_prop = topk_p

    decodedString = f"{prompt} {reverseNumeralize(decodedStrings[last_prop.argmax()], text_field)}"
    return decodedString

############################################################################################
# TASK 1.1
############################################################################################

def sample(model, text_field, prompt="", max_len=50, temp=1.0, k=0, p=1):
    assert (k==0 or p==1), "Cannot combine top-k and top-p sampling"

    output = text_field.process([text_field.tokenize(prompt.lower())]).to(dev)
    hidden_size = model.hidden_size
    num_layers = model.num_layers
    hidden_state = torch.zeros(num_layers, 1, hidden_size).to(dev)
    cell_state = torch.zeros(num_layers, 1, hidden_size).to(dev)

    crit = nn.Softmax(dim=1)
    strings=[]

    for _ in range(max_len):
        output, hidden_state, cell_state = model(output, hidden_state, cell_state)
        probs = crit(output[-1]/temp)

        if p > 0:
            sorted_p, sorted_idx = torch.sort(probs, descending=True)
            torch.use_deterministic_algorithms(False)
            cum_p = torch.cumsum(sorted_p, dim=1)
            torch.use_deterministic_algorithms(True)
            cum_p = cum_p.squeeze()

            idx_reset_probs = cum_p >= p            
            idx_reset_probs[0] = False 

            probs.squeeze()[sorted_idx[idx_reset_probs.unsqueeze(0)]] = 0
            probs = probs/probs.sum()
            distribution = torch.distributions.Categorical(probs)
            output = distribution.sample().unsqueeze(0)
        
        elif k != 0:
            topk_p, topk_idx = torch.topk(probs, k)
            topk_p = topk_p/topk_p.sum(1)
            distribution = torch.distributions.Categorical(topk_p)
            index_prop = distribution.sample()
            output = topk_idx.view(-1)[index_prop].unsqueeze(0)
            
        else:
            distribution = torch.distributions.Categorical(probs)
            output = distribution.sample().unsqueeze(0)

        strings.append(output)

    decodedString = f"{prompt} {reverseNumeralize(torch.cat(strings).squeeze(), text_field)}"
    return decodedString

############################################################################################

def reverseNumeralize(numeralized_string, text_field):
    strings = [text_field.vocab.itos[i] for i in numeralized_string]
    return " ".join(strings)

if __name__ == "__main__":
    main()
