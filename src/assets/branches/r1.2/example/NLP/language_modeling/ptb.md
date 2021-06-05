# Languge Modeling using LSTM on Penn Treebank

Language Modeling is the development of models to predict the next word of the sequence given the words that precede it. In this notebook we will demonstrate how to predict next word of a sequence using an LSTM. We will be using Penn Treebank dataset which contains 888K words for training, 70K for validation, and 79K for testing, with a vocabulary size of 10K.


```python
import tempfile

import fastestimator as fe
import numpy as np
import tensorflow as tf
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace import Trace
from fastestimator.trace.adapt import EarlyStopping, LRScheduler
from fastestimator.trace.io import BestModelSaver
```


```python
# Parameters
epochs=30
batch_size=128
seq_length=20
vocab_size=10000
data_dir=None
max_train_steps_per_epoch=None
save_dir=tempfile.mkdtemp()
```

## Building Components

### Downloading the data

First, we will download the Penn Treebank dataset via our dataset API.


```python
from fastestimator.dataset.data.penn_treebank import load_data
train_data, eval_data, _, vocab = load_data(root_dir=data_dir, seq_length=seq_length + 1)
```

### Step 1: Create `Pipeline`

We will create a custom NumpyOp to generate input and target sequences.


```python
class CreateInputAndTarget(NumpyOp):
    def forward(self, data, state):
        return data[:-1], data[1:]
```


```python
pipeline = fe.Pipeline(train_data=train_data,
                       eval_data=eval_data,
                       batch_size=batch_size,
                       ops=CreateInputAndTarget(inputs="x", outputs=("x", "y")),
                       drop_last=True)
```

### Step 2: Create `Network`

The architecture of our model is a LSTM.


```python
def build_model(vocab_size, embedding_dim, rnn_units, seq_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[None, seq_length]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model
```


```python
model = fe.build(model_fn=lambda: build_model(vocab_size, embedding_dim=300, rnn_units=600, seq_length=seq_length),
                     optimizer_fn=lambda: tf.optimizers.SGD(1.0, momentum=0.9))
```

We now define the `Network` object:


```python
network = fe.Network(ops=[
    ModelOp(model=model, inputs="x", outputs="y_pred"),
    CrossEntropy(
        inputs=("y_pred", "y"), outputs="ce", form="sparse", from_logits=True),
    UpdateOp(model=model, loss_name="ce")
])
```

In this example we will also use the following traces:

1. A custom trace to calculate Perplexity.
2. LRScheduler to apply custom learning rate schedule.
3. BestModelSaver for saving the best model. For illustration purpose, we will save these models in a temporary directory.
4. EarlyStopping Trace for stopping early.


```python
def lr_schedule(step, init_lr):
    if step <= 1725:
        lr = init_lr + init_lr * (step - 1) / 1725
    else:
        lr = max(2 * init_lr * ((6900 - step + 1725) / 6900), 1.0)
    return lr


class Perplexity(Trace):
    def on_epoch_end(self, data):
        ce = data["ce"]
        data.write_with_log(self.outputs[0], np.exp(ce))


traces = [
    Perplexity(inputs="ce", outputs="perplexity", mode="eval"),
    LRScheduler(model=model, lr_fn=lambda step: lr_schedule(step, init_lr=1.0)),
    BestModelSaver(model=model, save_dir=save_dir, metric='perplexity', save_best_mode='min', load_best_final=True),
    EarlyStopping(monitor="perplexity", patience=5)
]
```

### Step 3: Create `Estimator`


```python
estimator = fe.Estimator(pipeline=pipeline,
                         network=network,
                         epochs=epochs,
                         traces=traces,
                         max_train_steps_per_epoch=max_train_steps_per_epoch, 
                         log_steps=300)
```

## Training and Testing


```python
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; num_device: 1; logging_interval: 300; 
    FastEstimator-Train: step: 1; ce: 9.210202; model_lr: 1.0; 
    FastEstimator-Train: step: 300; ce: 6.110634; steps/sec: 8.33; model_lr: 1.1733333; 
    FastEstimator-Train: step: 345; epoch: 1; epoch_time: 43.55 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmptl5d8hgb/model_best_perplexity.h5
    FastEstimator-Eval: step: 345; epoch: 1; ce: 5.8996396; perplexity: 364.90594; since_best_perplexity: 0; min_perplexity: 364.90594; 
    FastEstimator-Train: step: 600; ce: 5.7039967; steps/sec: 8.27; model_lr: 1.3472464; 
    FastEstimator-Train: step: 690; epoch: 2; epoch_time: 41.65 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmptl5d8hgb/model_best_perplexity.h5
    FastEstimator-Eval: step: 690; epoch: 2; ce: 5.5457325; perplexity: 256.14215; since_best_perplexity: 0; min_perplexity: 256.14215; 
    FastEstimator-Train: step: 900; ce: 5.636613; steps/sec: 8.27; model_lr: 1.5211594; 
    FastEstimator-Train: step: 1035; epoch: 3; epoch_time: 41.89 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmptl5d8hgb/model_best_perplexity.h5
    FastEstimator-Eval: step: 1035; epoch: 3; ce: 5.3436885; perplexity: 209.28323; since_best_perplexity: 0; min_perplexity: 209.28323; 
    FastEstimator-Train: step: 1200; ce: 5.3110213; steps/sec: 8.23; model_lr: 1.6950724; 
    FastEstimator-Train: step: 1380; epoch: 4; epoch_time: 41.82 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmptl5d8hgb/model_best_perplexity.h5
    FastEstimator-Eval: step: 1380; epoch: 4; ce: 5.209713; perplexity: 183.04152; since_best_perplexity: 0; min_perplexity: 183.04152; 
    FastEstimator-Train: step: 1500; ce: 5.1398573; steps/sec: 8.25; model_lr: 1.8689855; 
    FastEstimator-Train: step: 1725; epoch: 5; epoch_time: 41.79 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmptl5d8hgb/model_best_perplexity.h5
    FastEstimator-Eval: step: 1725; epoch: 5; ce: 5.107191; perplexity: 165.20566; since_best_perplexity: 0; min_perplexity: 165.20566; 
    FastEstimator-Train: step: 1800; ce: 5.003286; steps/sec: 8.25; model_lr: 1.9782609; 
    FastEstimator-Train: step: 2070; epoch: 6; epoch_time: 41.9 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmptl5d8hgb/model_best_perplexity.h5
    FastEstimator-Eval: step: 2070; epoch: 6; ce: 5.018823; perplexity: 151.23322; since_best_perplexity: 0; min_perplexity: 151.23322; 
    FastEstimator-Train: step: 2100; ce: 4.9688864; steps/sec: 8.23; model_lr: 1.8913044; 
    FastEstimator-Train: step: 2400; ce: 4.8305387; steps/sec: 8.23; model_lr: 1.8043479; 
    FastEstimator-Train: step: 2415; epoch: 7; epoch_time: 41.97 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmptl5d8hgb/model_best_perplexity.h5
    FastEstimator-Eval: step: 2415; epoch: 7; ce: 4.9463377; perplexity: 140.65887; since_best_perplexity: 0; min_perplexity: 140.65887; 
    FastEstimator-Train: step: 2700; ce: 4.5900016; steps/sec: 8.23; model_lr: 1.7173913; 
    FastEstimator-Train: step: 2760; epoch: 8; epoch_time: 41.98 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmptl5d8hgb/model_best_perplexity.h5
    FastEstimator-Eval: step: 2760; epoch: 8; ce: 4.900966; perplexity: 134.41959; since_best_perplexity: 0; min_perplexity: 134.41959; 
    FastEstimator-Train: step: 3000; ce: 4.6566253; steps/sec: 8.21; model_lr: 1.6304348; 
    FastEstimator-Train: step: 3105; epoch: 9; epoch_time: 41.64 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmptl5d8hgb/model_best_perplexity.h5
    FastEstimator-Eval: step: 3105; epoch: 9; ce: 4.8612027; perplexity: 129.17947; since_best_perplexity: 0; min_perplexity: 129.17947; 
    FastEstimator-Train: step: 3300; ce: 4.6201677; steps/sec: 8.34; model_lr: 1.5434783; 
    FastEstimator-Train: step: 3450; epoch: 10; epoch_time: 41.66 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmptl5d8hgb/model_best_perplexity.h5
    FastEstimator-Eval: step: 3450; epoch: 10; ce: 4.833195; perplexity: 125.61168; since_best_perplexity: 0; min_perplexity: 125.61168; 
    FastEstimator-Train: step: 3600; ce: 4.672325; steps/sec: 8.27; model_lr: 1.4565217; 
    FastEstimator-Train: step: 3795; epoch: 11; epoch_time: 41.92 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmptl5d8hgb/model_best_perplexity.h5
    FastEstimator-Eval: step: 3795; epoch: 11; ce: 4.8110547; perplexity: 122.86113; since_best_perplexity: 0; min_perplexity: 122.86113; 
    FastEstimator-Train: step: 3900; ce: 4.5373406; steps/sec: 8.21; model_lr: 1.3695652; 
    FastEstimator-Train: step: 4140; epoch: 12; epoch_time: 41.83 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmptl5d8hgb/model_best_perplexity.h5
    FastEstimator-Eval: step: 4140; epoch: 12; ce: 4.802991; perplexity: 121.87438; since_best_perplexity: 0; min_perplexity: 121.87438; 
    FastEstimator-Train: step: 4200; ce: 4.412928; steps/sec: 8.26; model_lr: 1.2826087; 
    FastEstimator-Train: step: 4485; epoch: 13; epoch_time: 41.72 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmptl5d8hgb/model_best_perplexity.h5
    FastEstimator-Eval: step: 4485; epoch: 13; ce: 4.7911124; perplexity: 120.43527; since_best_perplexity: 0; min_perplexity: 120.43527; 
    FastEstimator-Train: step: 4500; ce: 4.402304; steps/sec: 8.26; model_lr: 1.1956521; 
    FastEstimator-Train: step: 4800; ce: 4.4627676; steps/sec: 8.31; model_lr: 1.1086956; 
    FastEstimator-Train: step: 4830; epoch: 14; epoch_time: 41.58 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmptl5d8hgb/model_best_perplexity.h5
    FastEstimator-Eval: step: 4830; epoch: 14; ce: 4.78295; perplexity: 119.45622; since_best_perplexity: 0; min_perplexity: 119.45622; 
    FastEstimator-Train: step: 5100; ce: 4.2155848; steps/sec: 8.24; model_lr: 1.0217391; 
    FastEstimator-Train: step: 5175; epoch: 15; epoch_time: 41.92 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmptl5d8hgb/model_best_perplexity.h5
    FastEstimator-Eval: step: 5175; epoch: 15; ce: 4.777499; perplexity: 118.80688; since_best_perplexity: 0; min_perplexity: 118.80688; 
    FastEstimator-Train: step: 5400; ce: 4.2096825; steps/sec: 8.24; model_lr: 1.0; 
    FastEstimator-Train: step: 5520; epoch: 16; epoch_time: 41.88 sec; 
    FastEstimator-Eval: step: 5520; epoch: 16; ce: 4.786487; perplexity: 119.87951; since_best_perplexity: 1; min_perplexity: 118.80688; 
    FastEstimator-Train: step: 5700; ce: 4.19374; steps/sec: 8.22; model_lr: 1.0; 
    FastEstimator-Train: step: 5865; epoch: 17; epoch_time: 41.86 sec; 
    FastEstimator-Eval: step: 5865; epoch: 17; ce: 4.791095; perplexity: 120.43314; since_best_perplexity: 2; min_perplexity: 118.80688; 
    FastEstimator-Train: step: 6000; ce: 4.040009; steps/sec: 8.27; model_lr: 1.0; 
    FastEstimator-Train: step: 6210; epoch: 18; epoch_time: 41.74 sec; 
    FastEstimator-Eval: step: 6210; epoch: 18; ce: 4.7963467; perplexity: 121.067314; since_best_perplexity: 3; min_perplexity: 118.80688; 
    FastEstimator-Train: step: 6300; ce: 4.0719166; steps/sec: 8.24; model_lr: 1.0; 
    FastEstimator-Train: step: 6555; epoch: 19; epoch_time: 41.94 sec; 
    FastEstimator-Eval: step: 6555; epoch: 19; ce: 4.799467; perplexity: 121.44568; since_best_perplexity: 4; min_perplexity: 118.80688; 
    FastEstimator-Train: step: 6600; ce: 4.110601; steps/sec: 8.24; model_lr: 1.0; 
    FastEstimator-Train: step: 6900; ce: 3.9709384; steps/sec: 8.3; model_lr: 1.0; 
    FastEstimator-Train: step: 6900; epoch: 20; epoch_time: 41.54 sec; 
    FastEstimator-EarlyStopping: 'perplexity' triggered an early stop. Its best value was 118.80687713623047 at epoch 15
    FastEstimator-Eval: step: 6900; epoch: 20; ce: 4.8052564; perplexity: 122.150795; since_best_perplexity: 5; min_perplexity: 118.80688; 
    FastEstimator-BestModelSaver: Restoring model from /tmp/tmptl5d8hgb/model_best_perplexity.h5
    FastEstimator-Finish: step: 6900; total_time: 864.19 sec; model_lr: 1.0; 


## Inferencing

Once the training is finished, we will use the model to generate some sequences of text.


```python
def get_next_word(data, vocab):
    output = network.transform(data, mode="infer") 
    index = output["y_pred"].numpy().squeeze()[-1].argmax()
    if index == 44:    # Removing unkwown predicition
        index = output["y_pred"].numpy().squeeze()[-1].argsort()[-2]
    return index

def generate_sequence(inp_seq, vocab, min_paragraph_len=50):
    data = pipeline.transform({"x": inp_seq}, mode="infer")
    generated_seq = data["x"]
    counter=0
    next_entry=0
    # Stopping at <eos> tag or after min_paragraph_len+30 words
    while (counter<min_paragraph_len or next_entry != 43) and counter<min_paragraph_len+30:  
        next_entry = get_next_word(data, vocab)
        generated_seq = np.concatenate([generated_seq.squeeze(), [next_entry]])
        data = {"x": generated_seq[-20:].reshape((1, 20))}
        counter+=1

    return " ".join([vocab[i] for i in generated_seq])
```

We will provide a text sequence from the validation dataset to the model and generate a paragraph with the input text sequence. 


```python
for _ in range(2):
    idx = np.random.choice(len(eval_data))
    inp_seq = eval_data["x"][idx]
    print("Input Sequence:", " ".join([vocab[i] for i in inp_seq[:20]]))
    gen_seq = generate_sequence(inp_seq, vocab, 50)
    print("\nGenerated Sequence:", gen_seq)
    print("\n")
```

    Input Sequence: the pictures <eos> the state <unk> noted that <unk> banking practices are grounds for removing an officer or director and
    
    Generated Sequence: the pictures <eos> the state <unk> noted that <unk> banking practices are grounds for removing an officer or director and chief executive officer <eos> mr. guber and mr. peters have been working on the board <eos> the company said the company will be able to pay for the $ N million of the company 's common shares outstanding <eos> the company said the company 's net income rose N N to $ N million from $ N million <eos>
    
    
    Input Sequence: the russians in iran the russians seem to have lost interest in the whole subject <eos> meanwhile congress is cutting
    
    Generated Sequence: the russians in iran the russians seem to have lost interest in the whole subject <eos> meanwhile congress is cutting the capital-gains tax cut to the u.s. and the u.s. trade deficit <eos> the u.s. trade deficit has been the highest since august N <eos> the dollar was mixed <eos> the dollar was mixed <eos> the nasdaq composite index rose N to N <eos> the index gained N to N <eos>
    
    


As you can see, the network is able to generate meaningful sentences.
