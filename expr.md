# Experiment trial

##  Download the dataset

    TODO: Understand the difficulty (size, length, typical SOTA result) of the following datasets

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

#### Shortened (in epoch) trial
+ `python -u main.py --epochs 5 --nlayers 3 --emsize 400 --nhid 1840 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.4 --wdrop 0.2 --wdecay 1.2e-6 --bptt 200 --batch_size 128 --optimizer adam --lr 1e-3 --data data/enwik8 --save ENWIK8.pt --when 25 35`

in ENWIK8-5.pt
test loss 1.15 

#### Vanilla
+ `python -u main.py --epochs 50 --nlayers 3 --emsize 400 --nhid 1840 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.4 --wdrop 0.2 --wdecay 1.2e-6 --bptt 200 --batch_size 128 --optimizer adam --lr 1e-3 --data data/enwik8 --save ENWIK8.pt --when 25 35`

in ENWIK8-50.pt
test loss 1.02
Somehow this code takes around 30 GB memory on GPU to run. 

### Character level Penn Treebank (PTB) with LSTM

#### Vanilla 

+ `python -u main.py --epochs 500 --nlayers 3 --emsize 200 --nhid 1000 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.25 --dropouti 0.1 --dropout 0.1 --wdrop 0.5 --wdecay 1.2e-6 --bptt 150 --batch_size 128 --optimizer adam --lr 2e-3 --data data/pennchar --save PTBC.pt --when 300 400`

test loss 
Somehow this code takes around TODO GB memory on GPU to run. 

After training, I don't keep the record, maybe I should make the output not only in terminal but also in record...

### Word level Penn Treebank (PTB) with LSTM

#### Vanilla 
+ `python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save PTB.pt`
+ `python finetune.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save PTB_finetuned.pt`
+ `python pointer.py --data data/penn --save PTB_pointed.pt --lambdasm 0.1 --theta 1.0 --window 500 --bptt 5000`

1. 2GB GPU memory (from aisci) to move PTB.pt, somehow it turned to 20GB occupation, somehow it out of memory...
    train loss test loss 
1.1 python main.py --batch_size 10 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save PTB_smaller_BS.pt
2. 
    train loss test loss 
3. 
    train loss test loss 

### Word level WikiText-2 (WT2) with LSTM


#### Vanilla 

+ `python main.py --epochs 750 --data data/wikitext-2 --save WT2.pt --dropouth 0.2 --seed 1882`
+ `python finetune.py --epochs 750 --data data/wikitext-2 --save WT2.pt --dropouth 0.2 --seed 1882`
+ `python pointer.py --save WT2.pt --lambdasm 0.1279 --theta 0.662 --window 3785 --bptt 2000 --data data/wikitext-2`

1.  GPU memory (from aisci) , somehow it turned to 20GB occupation
    train loss test loss 
2. 
    train loss test loss 
3. 
    train loss test loss 


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



## What is the transformer's performance on these datasets?






