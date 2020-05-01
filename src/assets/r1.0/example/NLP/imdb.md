<h1>IMDB Reviews sentiments prediction using LSTM</h1>


```python
import tempfile
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
import fastestimator as fe
from fastestimator.dataset.data import imdb_review
from fastestimator.op.numpyop.univariate.reshape import Reshape
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy
from fastestimator.backend import load_model
```


```python
MAX_WORDS = 10000
MAX_LEN = 500
batch_size = 64
epochs = 10
max_steps_per_epoch = None
```

<h2>Building components</h2>

<h3>Step 1: Prepare training & evaluation data and define Pipeline</h3>

We are loading the dataset from the tf.keras.datasets.imdb which contains movie reviews and sentiment scores. All the words have been replaced with the integers that specifies the popularity of the word in corpus. To ensure all the sequences are of same length we need to pad the input sequences before defining the Pipeline.


```python
train_data, eval_data = imdb_review.load_data(MAX_LEN, MAX_WORDS)
pipeline = fe.Pipeline(train_data=train_data,
                       eval_data=eval_data,
                       batch_size=batch_size,
                       ops=Reshape(1, inputs="y", outputs="y"))
```

<h3>Step 2: Create model and FastEstimator network</h3>

First, we have to define the network architecture and after defining the architecture, users are expected to feed the architecture definition, its associated model name and optimizer to fe.build.


```python
class ReviewSentiment(nn.Module):
    def __init__(self, embedding_size=64, hidden_units=64):
        super().__init__()
        self.embedding = nn.Embedding(MAX_WORDS, embedding_size)
        self.conv1d = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.maxpool1d = nn.MaxPool1d(kernel_size=4)
        self.lstm = nn.LSTM(input_size=125, hidden_size=hidden_units, num_layers=1)
        self.fc1 = nn.Linear(in_features=hidden_units, out_features=250)
        self.fc2 = nn.Linear(in_features=250, out_features=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute((0, 2, 1))
        x = self.conv1d(x)
        x = fn.relu(x)
        x = self.maxpool1d(x)
        output, _ = self.lstm(x)
        x = output[:, -1]  # sequence output of only last timestamp
        x = fn.tanh(x)
        x = self.fc1(x)
        x = fn.relu(x)
        x = self.fc2(x)
        x = fn.sigmoid(x)
        return x
```

Network is the object that define the whole logic of neural network, including models, loss functions, optimizers etc. A Network can have several different models and loss funcitons (like GAN). <b>fe.Network</b> takes series of operators and here we feed our model in the ModelOp with inputs and outputs. It should be noted that the y_pred is the key in the data dictionary which will store the predictions.


```python
model = fe.build(model_fn=lambda: ReviewSentiment(), optimizer_fn="adam")
network = fe.Network(ops=[
    ModelOp(model=model, inputs="x", outputs="y_pred"),
    CrossEntropy(inputs=("y_pred", "y"), outputs="loss"),
    UpdateOp(model=model, loss_name="loss")
])
```

<h3>Step 3: Prepare estimator and configure the training loop</h3>

<b>Estimator</b> is the API that wrap up the Pipeline, Network and other training metadata together. Estimator basically has four arguments network, pipeline, epochs and traces. Network and Pipeline objects are passed here as an argument. Traces are similar to the callbacks of Keras. The trace object will be called on specific timing during the training.

In the training loop, we want to measure the validation loss and save the model that has the minimum loss. BestModelSaver and Accuracy in the Trace class provide this convenient feature of storing the model.


```python
model_dir = tempfile.mkdtemp()
traces = [Accuracy(true_key="y", pred_key="y_pred"), BestModelSaver(model=model, save_dir=model_dir)]
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
                                                                            
    
    FastEstimator-Start: step: 1; model_lr: 0.001; 


    /home/ubuntu/anaconda3/envs/fe_env/lib/python3.6/site-packages/torch/nn/functional.py:1340: UserWarning: nn.functional.tanh is deprecated. Use torch.tanh instead.
      warnings.warn("nn.functional.tanh is deprecated. Use torch.tanh instead.")
    /home/ubuntu/anaconda3/envs/fe_env/lib/python3.6/site-packages/torch/nn/functional.py:1351: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
      warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")


    FastEstimator-Train: step: 1; loss: 0.6905144; 
    FastEstimator-Train: step: 100; loss: 0.69094294; steps/sec: 46.99; 
    FastEstimator-Train: step: 200; loss: 0.6621749; steps/sec: 48.85; 
    FastEstimator-Train: step: 300; loss: 0.5835465; steps/sec: 54.12; 
    FastEstimator-Train: step: 391; epoch: 1; epoch_time: 7.74 sec; 
    Saved model to /tmp/tmp69qyfzvm/model_best_loss.pt
    FastEstimator-Eval: step: 391; epoch: 1; loss: 0.5250161; min_loss: 0.5250161; since_best: 0; accuracy: 0.7377667446412374; 
    FastEstimator-Train: step: 400; loss: 0.51533854; steps/sec: 55.54; 
    FastEstimator-Train: step: 500; loss: 0.6381638; steps/sec: 60.01; 
    FastEstimator-Train: step: 600; loss: 0.4390931; steps/sec: 58.14; 
    FastEstimator-Train: step: 700; loss: 0.32808638; steps/sec: 59.57; 
    FastEstimator-Train: step: 782; epoch: 2; epoch_time: 6.7 sec; 
    Saved model to /tmp/tmp69qyfzvm/model_best_loss.pt
    FastEstimator-Eval: step: 782; epoch: 2; loss: 0.41990075; min_loss: 0.41990075; since_best: 0; accuracy: 0.8072277653124552; 
    FastEstimator-Train: step: 800; loss: 0.4495564; steps/sec: 56.01; 
    FastEstimator-Train: step: 900; loss: 0.38001418; steps/sec: 60.14; 
    FastEstimator-Train: step: 1000; loss: 0.28246647; steps/sec: 60.33; 
    FastEstimator-Train: step: 1100; loss: 0.36126548; steps/sec: 60.51; 
    FastEstimator-Train: step: 1173; epoch: 3; epoch_time: 6.63 sec; 
    Saved model to /tmp/tmp69qyfzvm/model_best_loss.pt
    FastEstimator-Eval: step: 1173; epoch: 3; loss: 0.39232534; min_loss: 0.39232534; since_best: 0; accuracy: 0.8241752995655702; 
    FastEstimator-Train: step: 1200; loss: 0.32620478; steps/sec: 55.57; 
    FastEstimator-Train: step: 1300; loss: 0.33430642; steps/sec: 60.1; 
    FastEstimator-Train: step: 1400; loss: 0.21134894; steps/sec: 62.23; 
    FastEstimator-Train: step: 1500; loss: 0.34480703; steps/sec: 62.4; 
    FastEstimator-Train: step: 1564; epoch: 4; epoch_time: 6.47 sec; 
    FastEstimator-Eval: step: 1564; epoch: 4; loss: 0.3997118; min_loss: 0.39232534; since_best: 1; accuracy: 0.8274693273499785; 
    FastEstimator-Train: step: 1600; loss: 0.14769143; steps/sec: 57.72; 
    FastEstimator-Train: step: 1700; loss: 0.17477548; steps/sec: 60.4; 
    FastEstimator-Train: step: 1800; loss: 0.34234992; steps/sec: 60.82; 
    FastEstimator-Train: step: 1900; loss: 0.34789586; steps/sec: 61.12; 
    FastEstimator-Train: step: 1955; epoch: 5; epoch_time: 6.55 sec; 
    FastEstimator-Eval: step: 1955; epoch: 5; loss: 0.39978975; min_loss: 0.39232534; since_best: 2; accuracy: 0.8300950016708837; 
    FastEstimator-Train: step: 2000; loss: 0.21192178; steps/sec: 56.29; 
    FastEstimator-Train: step: 2100; loss: 0.24565384; steps/sec: 60.85; 
    FastEstimator-Train: step: 2200; loss: 0.21373041; steps/sec: 60.08; 
    FastEstimator-Train: step: 2300; loss: 0.24357724; steps/sec: 61.3; 
    FastEstimator-Train: step: 2346; epoch: 6; epoch_time: 6.57 sec; 
    FastEstimator-Eval: step: 2346; epoch: 6; loss: 0.39285892; min_loss: 0.39232534; since_best: 3; accuracy: 0.8357282665775528; 
    FastEstimator-Train: step: 2400; loss: 0.08471558; steps/sec: 56.66; 
    FastEstimator-Train: step: 2500; loss: 0.20877948; steps/sec: 60.78; 
    FastEstimator-Train: step: 2600; loss: 0.09914401; steps/sec: 61.27; 
    FastEstimator-Train: step: 2700; loss: 0.15458922; steps/sec: 60.94; 
    FastEstimator-Train: step: 2737; epoch: 7; epoch_time: 6.53 sec; 
    FastEstimator-Eval: step: 2737; epoch: 7; loss: 0.43396476; min_loss: 0.39232534; since_best: 4; accuracy: 0.8326729364586815; 
    FastEstimator-Train: step: 2800; loss: 0.16790089; steps/sec: 56.49; 
    FastEstimator-Train: step: 2900; loss: 0.09017545; steps/sec: 60.99; 
    FastEstimator-Train: step: 3000; loss: 0.06604583; steps/sec: 61.62; 
    FastEstimator-Train: step: 3100; loss: 0.2692815; steps/sec: 61.39; 
    FastEstimator-Train: step: 3128; epoch: 8; epoch_time: 6.52 sec; 
    FastEstimator-Eval: step: 3128; epoch: 8; loss: 0.47958812; min_loss: 0.39232534; since_best: 5; accuracy: 0.8260371413567575; 
    FastEstimator-Train: step: 3200; loss: 0.12287539; steps/sec: 56.58; 
    FastEstimator-Train: step: 3300; loss: 0.16652104; steps/sec: 62.04; 
    FastEstimator-Train: step: 3400; loss: 0.08797251; steps/sec: 59.71; 
    FastEstimator-Train: step: 3500; loss: 0.08476864; steps/sec: 60.8; 
    FastEstimator-Train: step: 3519; epoch: 9; epoch_time: 6.56 sec; 
    FastEstimator-Eval: step: 3519; epoch: 9; loss: 0.51876205; min_loss: 0.39232534; since_best: 6; accuracy: 0.824270778631785; 
    FastEstimator-Train: step: 3600; loss: 0.14876236; steps/sec: 56.08; 
    FastEstimator-Train: step: 3700; loss: 0.11151114; steps/sec: 60.72; 
    FastEstimator-Train: step: 3800; loss: 0.05140955; steps/sec: 60.18; 
    FastEstimator-Train: step: 3900; loss: 0.062084932; steps/sec: 59.42; 
    FastEstimator-Train: step: 3910; epoch: 10; epoch_time: 6.62 sec; 
    FastEstimator-Eval: step: 3910; epoch: 10; loss: 0.5560739; min_loss: 0.39232534; since_best: 7; accuracy: 0.831049792333031; 
    FastEstimator-Finish: step: 3910; total_time: 87.54 sec; model_lr: 0.001; 


<h2>Inferencing</h2>

For inferencing, first we have to load the trained model weights. We saved model weights with minimum loss and we will load the weights using <i>load_model</i>


```python
model_name = 'model_best_loss.pt'
model_path = os.path.join(model_dir, model_name)
load_model(model, model_path)
```

    Loaded model weights from /tmp/tmp69qyfzvm/model_best_loss.pt


Get any random sequence and compare the prediction with the ground truth.


```python
selected_idx = np.random.randint(10000)
print("Ground truth is: ",eval_data[selected_idx]['y'])
```

    Ground truth is:  0


Create data dictionary for the inference. <i>Transform()</i> function in Pipeline and Network applies all the operations on the given data.


```python
infer_data = {"x":eval_data[selected_idx]['x'], "y":eval_data[selected_idx]['y']}
data = pipeline.transform(infer_data, mode="infer")
data = network.transform(data, mode="infer")
```

Finally, print the inferencing results.


```python
print("Prediction for the input sequence: ", np.array(data["y_pred"])[0][0])
```

    Prediction for the input sequence:  0.30634004

