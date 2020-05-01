<h1>Named Entity Recognition using BERT Fine tuning</h1>

For the downstream NLP tasks such as question answering, named entity recognition, and language inference, pre-trained word representations tend to perform better. BERT which fine tunes deep bi-directional representation on series of tasks achieves state-of-the-art results. Unlike traditional Tranformer, BERT is trained on “masked language modeling,” which means that it is allowed to see the whole sentence and does not limit the context it can take into account.

For this example, we are leveraging transformers library to load BERT model and other config files


```python
import tempfile
import os
import numpy as np
from typing import Callable, Iterable, List, Union

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from transformers import BertTokenizer, TFBertModel

import fastestimator as fe
from fastestimator.dataset.data import german_ner
from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.op.numpyop.univariate import PadSequence, Tokenize, WordtoId
from fastestimator.op.tensorop import TensorOp, Reshape
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy
from fastestimator.trace.io import BestModelSaver
from fastestimator.backend import feed_forward
```


```python
max_len = 20
batch_size = 64
epochs = 10
max_steps_per_epoch = None
save_dir = tempfile.mkdtemp()
data_dir = None
```

Custom NumpyOp that constructs attention masks for input sequences


```python
class AttentionMask(NumpyOp):
    def forward(self, data, state):
        masks = [float(i > 0) for i in data]
        return np.array(masks)
```

char2idx function creates look-up table for the corresponding ids for the labels


```python
def char2idx(data):
    tag2idx = {t: i for i, t in enumerate(data)}
    return tag2idx
```

<h2>Building components</h2>

<h3>Step 1: Prepare training & evaluation data and define pipeline</h3>

NER dataset from GermEval contains sequences and entity tags from german wikipedia and news corpora. We are loading train and eval sequences dataset along with data and label vocabulary. For this example other nouns are omitted for the simplicity.


```python
train_data, eval_data, data_vocab, label_vocab = german_ner.load_data(root_dir=data_dir)
```

Define a pipeline to tokenize and pad the input sequences and construct attention masks. Attention masks are used to avoid performing attention operation on padded tokens. We are using BERT tokenizer for input sequences tokenization and max length 50 for this example.


```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tag2idx = char2idx(label_vocab)
pipeline = fe.Pipeline(
    train_data=train_data,
    eval_data=eval_data,
    batch_size=batch_size,
    ops=[
        Tokenize(inputs="x", outputs="x", tokenize_fn=tokenizer.tokenize),
        WordtoId(inputs="x", outputs="x", mapping=tokenizer.convert_tokens_to_ids),
        WordtoId(inputs="y", outputs="y", mapping=tag2idx),
        PadSequence(max_len=max_len, inputs="x", outputs="x"),
        PadSequence(max_len=max_len, value=len(tag2idx), inputs="y", outputs="y"),
        AttentionMask(inputs="x", outputs="x_masks")
    ])
```

<h3>Step 2: Create model and FastEstimator network</h3>

Network architecture has pretrained weights as initialization for downsteam task. Whole network is then trained during the fine tuning.


```python
def ner_model():
    token_inputs = Input((max_len), dtype=tf.int32, name='input_words')
    mask_inputs = Input((max_len), dtype=tf.int32, name='input_masks')
    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    seq_output, _ = bert_model(token_inputs, attention_mask=mask_inputs)
    output = Dense(24, activation='softmax')(seq_output)
    model = Model([token_inputs, mask_inputs], output)
    return model
```

Model definition is then intantiated by calling fe.build which also associates the model with specific optimizers.


```python
model = fe.build(model_fn=ner_model, optimizer_fn=lambda: tf.optimizers.Adam(1e-5))
```

fe.Network takes series of operators and here we feed our model in the ModelOp with inputs and outputs. Here, ReshapeOp transforms the prediction and ground truth to scalar or two dimensional vector before feeding it to loss calculation.


```python
network = fe.Network(ops=[
        ModelOp(model=model, inputs=["x", "x_masks"], outputs="y_pred"),
        Reshape(inputs="y", outputs="y", shape=(-1, )),
        Reshape(inputs="y_pred", outputs="y_pred", shape=(-1, 24)),
        CrossEntropy(inputs=("y_pred", "y"), outputs="loss"),
        UpdateOp(model=model, loss_name="loss")
    ])
```

<h3>Step 3: Prepare Estimator and configure the training loop</h3>

Estimator basically has four arguments network, pipeline, epochs and traces. During the training, we want the accuracy metric and save the model with minimum loss, we will define that in Trace class.


```python
traces = [Accuracy(true_key="y", pred_key="y_pred"), BestModelSaver(model=model, save_dir=save_dir)]
```


```python
estimator = fe.Estimator(network=network,
                         pipeline=pipeline,
                         epochs=epochs,
                         traces=traces, 
                         max_steps_per_epoch=max_steps_per_epoch)
```

<h2>Training</h2>


```python
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; model_lr: 1e-05; 
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_model/bert/pooler/dense/kernel:0', 'tf_bert_model/bert/pooler/dense/bias:0'] when minimizing the loss.
    FastEstimator-Train: step: 1; loss: 3.8005962; 
    FastEstimator-Train: step: 100; loss: 0.40420213; steps/sec: 2.05; 
    FastEstimator-Train: step: 125; epoch: 1; epoch_time: 72.91 sec; 
    FastEstimator-ModelSaver: saved model to /tmp/tmpk1i5vjc2/model_best_loss.h5
    FastEstimator-Eval: step: 125; epoch: 1; loss: 0.30054897; min_loss: 0.30054897; since_best: 0; accuracy: 0.9269; 
    FastEstimator-Train: step: 200; loss: 0.22695072; steps/sec: 2.0; 
    FastEstimator-Train: step: 250; epoch: 2; epoch_time: 62.35 sec; 
    FastEstimator-ModelSaver: saved model to /tmp/tmpk1i5vjc2/model_best_loss.h5
    FastEstimator-Eval: step: 250; epoch: 2; loss: 0.1562941; min_loss: 0.1562941; since_best: 0; accuracy: 0.947175; 
    FastEstimator-Train: step: 300; loss: 0.1829088; steps/sec: 1.99; 
    FastEstimator-Train: step: 375; epoch: 3; epoch_time: 62.8 sec; 
    FastEstimator-ModelSaver: saved model to /tmp/tmpk1i5vjc2/model_best_loss.h5
    FastEstimator-Eval: step: 375; epoch: 3; loss: 0.12876438; min_loss: 0.12876438; since_best: 0; accuracy: 0.956675; 
    FastEstimator-Train: step: 400; loss: 0.1481853; steps/sec: 2.0; 
    FastEstimator-Train: step: 500; loss: 0.13613644; steps/sec: 2.01; 
    FastEstimator-Train: step: 500; epoch: 4; epoch_time: 62.44 sec; 
    FastEstimator-ModelSaver: saved model to /tmp/tmpk1i5vjc2/model_best_loss.h5
    FastEstimator-Eval: step: 500; epoch: 4; loss: 0.10898326; min_loss: 0.10898326; since_best: 0; accuracy: 0.962875; 
    FastEstimator-Train: step: 600; loss: 0.12551221; steps/sec: 2.0; 
    FastEstimator-Train: step: 625; epoch: 5; epoch_time: 62.37 sec; 
    FastEstimator-ModelSaver: saved model to /tmp/tmpk1i5vjc2/model_best_loss.h5
    FastEstimator-Eval: step: 625; epoch: 5; loss: 0.097521596; min_loss: 0.097521596; since_best: 0; accuracy: 0.966675; 
    FastEstimator-Train: step: 700; loss: 0.11037835; steps/sec: 2.0; 
    FastEstimator-Train: step: 750; epoch: 6; epoch_time: 62.53 sec; 
    FastEstimator-ModelSaver: saved model to /tmp/tmpk1i5vjc2/model_best_loss.h5
    FastEstimator-Eval: step: 750; epoch: 6; loss: 0.0885827; min_loss: 0.0885827; since_best: 0; accuracy: 0.970525; 
    FastEstimator-Train: step: 800; loss: 0.09738168; steps/sec: 2.0; 
    FastEstimator-Train: step: 875; epoch: 7; epoch_time: 62.47 sec; 
    FastEstimator-ModelSaver: saved model to /tmp/tmpk1i5vjc2/model_best_loss.h5
    FastEstimator-Eval: step: 875; epoch: 7; loss: 0.08195208; min_loss: 0.08195208; since_best: 0; accuracy: 0.97235; 
    FastEstimator-Train: step: 900; loss: 0.11297427; steps/sec: 2.0; 
    FastEstimator-Train: step: 1000; loss: 0.08556217; steps/sec: 2.01; 
    FastEstimator-Train: step: 1000; epoch: 8; epoch_time: 62.44 sec; 
    FastEstimator-ModelSaver: saved model to /tmp/tmpk1i5vjc2/model_best_loss.h5
    FastEstimator-Eval: step: 1000; epoch: 8; loss: 0.07855953; min_loss: 0.07855953; since_best: 0; accuracy: 0.974875; 
    FastEstimator-Train: step: 1100; loss: 0.108659945; steps/sec: 2.01; 
    FastEstimator-Train: step: 1125; epoch: 9; epoch_time: 62.28 sec; 
    FastEstimator-ModelSaver: saved model to /tmp/tmpk1i5vjc2/model_best_loss.h5
    FastEstimator-Eval: step: 1125; epoch: 9; loss: 0.074070126; min_loss: 0.074070126; since_best: 0; accuracy: 0.9765; 
    FastEstimator-Train: step: 1200; loss: 0.071078494; steps/sec: 2.0; 
    FastEstimator-Train: step: 1250; epoch: 10; epoch_time: 62.34 sec; 
    FastEstimator-ModelSaver: saved model to /tmp/tmpk1i5vjc2/model_best_loss.h5
    FastEstimator-Eval: step: 1250; epoch: 10; loss: 0.069807574; min_loss: 0.069807574; since_best: 0; accuracy: 0.977925; 
    FastEstimator-Finish: step: 1250; total_time: 728.17 sec; model_lr: 1e-05; 


<h2>Inferencing</h2>

Load model weights using <i>fe.build</i>


```python
model_name = 'model_best_loss.h5'
model_path = os.path.join(save_dir, model_name)
trained_model = fe.build(model_fn=ner_model, weights_path=model_path, optimizer_fn=lambda: tf.optimizers.Adam(1e-5))
```

    Loaded model weights from /tmp/tmpk1i5vjc2/model_best_loss.h5



```python
selected_idx = np.random.randint(1000)
print("Ground truth is: ",eval_data[selected_idx]['y'])
```

    Ground truth is:  ['B-PER', 'I-PER', 'I-PER', 'I-PER']


Create data dictionary for the inference. Transform() function in Pipeline and Network applies all the operations on the given data.


```python
infer_data = {"x":eval_data[selected_idx]['x'], "y":eval_data[selected_idx]['y']}
data = pipeline.transform(infer_data, mode="infer")
data = network.transform(data, mode="infer")
```

Get the predictions using <i>feed_forward</i>


```python
predictions = feed_forward(trained_model, [data["x"],data["x_masks"]], training=False)
predictions = np.array(predictions).reshape(20,24)
predictions = np.argmax(predictions, axis=-1)
```


```python
def get_key(val): 
    for key, value in tag2idx.items(): 
         if val == value: 
            return key 
```


```python
print("Predictions: ", [get_key(pred) for pred in predictions])
```

    Predictions:  ['B-PER', 'I-PER', 'I-PER', 'I-PER', None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

