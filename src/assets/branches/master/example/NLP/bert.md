
# Named Entity Recognition using BERT Fine-Tuning

For downstream NLP tasks such as question answering, named entity recognition, and language inference, models built on pre-trained word representations tend to perform better. BERT, which fine tunes a deep bi-directional representation on a series of tasks, achieves state-of-the-art results. Unlike traditional transformers, BERT is trained on "masked language modeling," which means that it is allowed to see the whole sentence and does not limit the context it can take into account.

For this example, we are leveraging the transformers library to load a BERT model, along with some config files:


```python
import tempfile
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from transformers import BertTokenizer, TFBertModel

import fastestimator as fe
from fastestimator.dataset.data import mitmovie_ner
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
max_train_steps_per_epoch = None
max_eval_steps_per_epoch = None
save_dir = tempfile.mkdtemp()
data_dir = None
```

We will need a custom `NumpyOp` that constructs attention masks for input sequences:


```python
class AttentionMask(NumpyOp):
    def forward(self, data, state):
        masks = [float(i > 0) for i in data]
        return np.array(masks)
```

Our `char2idx` function creates a look-up table to match ids and labels:


```python
def char2idx(data):
    tag2idx = {t: i for i, t in enumerate(data)}
    return tag2idx
```

<h2>Building components</h2>

### Step 1: Prepare training & evaluation data and define a `Pipeline`

We are loading train and eval sequences from the MIT Movie datasets that is semantically tagged along with data and label vocabulary. For this example other nouns are omitted for the simplicity.


```python
train_data, eval_data, data_vocab, label_vocab = mitmovie_ner.load_data(root_dir=data_dir)
```

    /home/ubuntu/fe/fastestimator/fastestimator/dataset/data/mitmovie_ner.py:101: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
      y_train = np.array(y_train)
    /home/ubuntu/fe/fastestimator/fastestimator/dataset/data/mitmovie_ner.py:102: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
      y_eval = np.array(y_eval)


Define a pipeline to tokenize and pad the input sequences and construct attention masks. Attention masks are used to avoid performing attention operations on padded tokens. We are using the BERT tokenizer for input sequence tokenization, and limiting our sequences to a max length of 50 for this example.


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

### Step 2: Create `model` and FastEstimator `Network`

Our neural network architecture leverages pre-trained weights as initialization for downstream tasks. The whole network is then trained during the fine-tuning.


```python
def ner_model():
    token_inputs = Input((max_len), dtype=tf.int32, name='input_words')
    mask_inputs = Input((max_len), dtype=tf.int32, name='input_masks')
    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    seq_output, _ = bert_model(token_inputs, attention_mask=mask_inputs)
    output = Dense(len(label_vocab) + 1, activation='softmax')(seq_output)
    model = Model([token_inputs, mask_inputs], output)
    return model
```

After defining the model, it is then instantiated by calling fe.build which also associates the model with a specific optimizer:


```python
model = fe.build(model_fn=ner_model, optimizer_fn=lambda: tf.optimizers.Adam(1e-5))
```

`fe.Network` takes a series of operators. In this case we use a `ModelOp` to run forward passes through the neural network. The `ReshapeOp` is then used to transform the prediction and ground truth to a two dimensional vector or scalar respectively before feeding them to the loss calculation.


```python
network = fe.Network(ops=[
        ModelOp(model=model, inputs=["x", "x_masks"], outputs="y_pred"),
        Reshape(inputs="y", outputs="y", shape=(-1, )),
        Reshape(inputs="y_pred", outputs="y_pred", shape=(-1, len(label_vocab) + 1)),
        CrossEntropy(inputs=("y_pred", "y"), outputs="loss"),
        UpdateOp(model=model, loss_name="loss")
    ])
```

### Step 3: Prepare `Estimator` and configure the training loop

The `Estimator` takes four important arguments: network, pipeline, epochs, and traces. During the training, we want to compute accuracy as well as to save the model with the minimum loss. This can be done using `Traces`.


```python
traces = [Accuracy(true_key="y", pred_key="y_pred"), BestModelSaver(model=model, save_dir=save_dir)]
```


```python
estimator = fe.Estimator(network=network,
                         pipeline=pipeline,
                         epochs=epochs,
                         traces=traces, 
                         max_train_steps_per_epoch=max_train_steps_per_epoch,
                         max_eval_steps_per_epoch=max_eval_steps_per_epoch)
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
                                                                            
    
    FastEstimator-Start: step: 1; num_device: 1; logging_interval: 100; 
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_model/bert/pooler/dense/kernel:0', 'tf_bert_model/bert/pooler/dense/bias:0'] when minimizing the loss.
    FastEstimator-Train: step: 1; loss: 3.534539; 
    FastEstimator-Train: step: 100; loss: 0.7910271; steps/sec: 2.04; 
    WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_model/bert/pooler/dense/kernel:0', 'tf_bert_model/bert/pooler/dense/bias:0'] when minimizing the loss.
    FastEstimator-Train: step: 153; epoch: 1; epoch_time: 93.27 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpjyf_2m8t/model_best_loss.h5
    FastEstimator-Eval: step: 153; epoch: 1; loss: 0.4999621; accuracy: 0.8571633237822349; since_best_loss: 0; min_loss: 0.4999621; 
    FastEstimator-Train: step: 200; loss: 0.4584582; steps/sec: 1.71; 
    FastEstimator-Train: step: 300; loss: 0.34510213; steps/sec: 1.97; 
    FastEstimator-Train: step: 306; epoch: 2; epoch_time: 77.24 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpjyf_2m8t/model_best_loss.h5
    FastEstimator-Eval: step: 306; epoch: 2; loss: 0.33649036; accuracy: 0.9036635284486287; since_best_loss: 0; min_loss: 0.33649036; 
    FastEstimator-Train: step: 400; loss: 0.28147238; steps/sec: 1.89; 
    FastEstimator-Train: step: 459; epoch: 3; epoch_time: 81.27 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpjyf_2m8t/model_best_loss.h5
    FastEstimator-Eval: step: 459; epoch: 3; loss: 0.27175826; accuracy: 0.920384772820303; since_best_loss: 0; min_loss: 0.27175826; 
    FastEstimator-Train: step: 500; loss: 0.2756353; steps/sec: 1.92; 
    FastEstimator-Train: step: 600; loss: 0.22491762; steps/sec: 2.0; 
    FastEstimator-Train: step: 612; epoch: 4; epoch_time: 76.74 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpjyf_2m8t/model_best_loss.h5
    FastEstimator-Eval: step: 612; epoch: 4; loss: 0.23077382; accuracy: 0.9328694228407696; since_best_loss: 0; min_loss: 0.23077382; 
    FastEstimator-Train: step: 700; loss: 0.21992946; steps/sec: 2.0; 
    FastEstimator-Train: step: 765; epoch: 5; epoch_time: 76.65 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpjyf_2m8t/model_best_loss.h5
    FastEstimator-Eval: step: 765; epoch: 5; loss: 0.20521004; accuracy: 0.9419566107245191; since_best_loss: 0; min_loss: 0.20521004; 
    FastEstimator-Train: step: 800; loss: 0.22478268; steps/sec: 2.0; 
    FastEstimator-Train: step: 900; loss: 0.18137912; steps/sec: 1.99; 
    FastEstimator-Train: step: 918; epoch: 6; epoch_time: 76.67 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpjyf_2m8t/model_best_loss.h5
    FastEstimator-Eval: step: 918; epoch: 6; loss: 0.19312857; accuracy: 0.947359803520262; since_best_loss: 0; min_loss: 0.19312857; 
    FastEstimator-Train: step: 1000; loss: 0.16234711; steps/sec: 1.99; 
    FastEstimator-Train: step: 1071; epoch: 7; epoch_time: 76.88 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpjyf_2m8t/model_best_loss.h5
    FastEstimator-Eval: step: 1071; epoch: 7; loss: 0.17946187; accuracy: 0.951105198526402; since_best_loss: 0; min_loss: 0.17946187; 
    FastEstimator-Train: step: 1100; loss: 0.1411412; steps/sec: 1.99; 
    FastEstimator-Train: step: 1200; loss: 0.21398099; steps/sec: 2.0; 
    FastEstimator-Train: step: 1224; epoch: 8; epoch_time: 76.62 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpjyf_2m8t/model_best_loss.h5
    FastEstimator-Eval: step: 1224; epoch: 8; loss: 0.17292295; accuracy: 0.9533974621367172; since_best_loss: 0; min_loss: 0.17292295; 
    FastEstimator-Train: step: 1300; loss: 0.104645446; steps/sec: 1.98; 
    FastEstimator-Train: step: 1377; epoch: 9; epoch_time: 77.22 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpjyf_2m8t/model_best_loss.h5
    FastEstimator-Eval: step: 1377; epoch: 9; loss: 0.17160487; accuracy: 0.95440032746623; since_best_loss: 0; min_loss: 0.17160487; 
    FastEstimator-Train: step: 1400; loss: 0.114030465; steps/sec: 1.99; 
    FastEstimator-Train: step: 1500; loss: 0.118343726; steps/sec: 2.0; 
    FastEstimator-Train: step: 1530; epoch: 10; epoch_time: 76.59 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpjyf_2m8t/model_best_loss.h5
    FastEstimator-Eval: step: 1530; epoch: 10; loss: 0.16745299; accuracy: 0.9565083913221449; since_best_loss: 0; min_loss: 0.16745299; 
    FastEstimator-Finish: step: 1530; total_time: 897.96 sec; model_lr: 1e-05; 


<h2>Inferencing</h2>

Load model weights using <i>fe.build</i>


```python
model_name = 'model_best_loss.h5'
model_path = os.path.join(save_dir, model_name)
trained_model = fe.build(model_fn=ner_model, weights_path=model_path, optimizer_fn=lambda: tf.optimizers.Adam(1e-5))
```

Let's take random phrase about a movie and predict it's named entities in BIO format.


```python
test_input = 'have you seen The dark night trilogy'
test_ground_truth = ['O', 'O', 'O', 'O', 'B-TITLE', 'I-TITLE', 'I-TITLE']
```

Create a data dictionary for the inference. The `transform()` function in `Pipeline` and `Network` applies all their operations on the given data:


```python
infer_data = {"x":test_input, "y":test_ground_truth}
data = pipeline.transform(infer_data, mode="infer")
data = network.transform(data, mode="infer")
```

Get the predictions using <i>feed_forward</i>


```python
predictions = feed_forward(trained_model, [data["x"],data["x_masks"]], training=False)
predictions = np.array(predictions).reshape(20, len(label_vocab) + 1)
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

    Predictions:  ['O', 'O', 'O', 'O', 'B-TITLE', 'I-TITLE', 'I-TITLE', None, None, None, None, None, None, None, None, None, None, None, None, None]

