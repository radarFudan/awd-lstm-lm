# Experiment trial

##  Download the dataset
    
    1. WikiText-2 (WT2)
    2. WikiText-103 (WT103)
    3. enwik8 (Character)
        character level language models over the Penn Treebank (PTBC) and Hutter Prize dataset
    4. Penn Treebank (PTB)
        word level language models

## Prepare the dataset

    `awd-lstm-lm/data/enwik8/prep_enwik8.py`
    No need, done in data downloading.

## Code reading

    In `utils.py/get_batch`, it seems the sequence length is limited to bptt. 
    
    Question: What is the default sequence length in vanilla RNN, for example torch. 

## Running and debugging

### Character level enwik8 with LSTM

Shortened (in epoch) trial
+ `python -u main.py --epochs 5 --nlayers 3 --emsize 400 --nhid 1840 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.4 --wdrop 0.2 --wdecay 1.2e-6 --bptt 200 --batch_size 128 --optimizer adam --lr 1e-3 --data data/enwik8 --save ENWIK8.pt --when 25 35`

epoch 1 2600/3515, loss 1.39, ms/batch 122.49 ppl 3.95, bpc 2.0

### What is ppl, bpc?

PPL, perplexity

In information theory, perplexity is a measurement of how well a probability distribution or probability model predicts a sample. It may be used to compare probability models. A low perplexity indicates the probability distribution is good at predicting the sample.

$ppl = 2^{cross\ entropy}, bpc = \log_2(Perplexity)$


### what’s the component that make it works?

LSTM, 
dropout on input, hidden, output
SplitCrossEntropyLoss calculates an approximate softmax, don't understand
weight drop, - regularization

what's the purpose of this pointer code? 
In pointer, there is bptt, which is the sequence length. 
However, there is also a pointer window length, what's this? 

What is this theta, and the following lambda sm
parser.add_argument(
    "--theta",
    type=float,
    default=0.6625523432485668,
    help="mix between uniform distribution and pointer softmax distribution over previous words",
)
parser.add_argument(
    "--lambdasm",
    type=float,
    default=0.12785920428335693,
    help="linear mix between only pointer (1) and only vocab (0) distribution",
)




