# IMDB Reviews Sentiments prediction with LSTM


```python
import os
import warnings
import numpy as np
import tensorflow as tf
import tempfile
import fastestimator as fe

from tensorflow.python.keras import layers
from fastestimator.op.tensorop import BinaryCrossentropy, ModelOp, Reshape
from fastestimator.trace import Accuracy, ModelSaver
warnings.filterwarnings('ignore')
```

For this example we are defining the vocabulary size to 10000 and maximum sequence length to 500. That can also be changed later on as hyperparameters.


```python
MAX_WORDS = 10000
MAX_LEN = 500
batch_size = 64
epochs = 10
steps_per_epoch=None
validation_steps=None
```

## Step 1 : Prepare training and evaluation dataset, define FastEstimator Pipeline

We are loading the dataset from the tf.keras.datasets.imdb which contains movie reviews and sentiment scores.
All the words have been replaced with the integers that specifies the popularity of the word in corpus. To ensure all the sequences are of same length we need to pad the input sequences before defining the Pipeline.


```python
def pad(input_list, padding_size, padding_value):
    return input_list + [padding_value] * abs((len(input_list) - padding_size))
```


```python
#load the data and pad the sequences
(x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.imdb.load_data(maxlen=MAX_LEN, num_words=MAX_WORDS)
data = {
        "train": {
            "x": np.array([pad(x, MAX_LEN, 0) for x in x_train]),
            "y": y_train
        },
        "eval": {
            "x": np.array([pad(x, MAX_LEN, 0) for x in x_eval]),
            "y": y_eval
        }
    }
```

Now, we can define the Pipeline passing the batch size and data dictionary. We need to reshape the groud truth from (batch_size,) to (batch_size, 1) 


```python
pipeline = fe.Pipeline(batch_size=batch_size, data=data, ops=Reshape([1], inputs="y", outputs="y"))
```

## Step 2: Create model and FastEstimator Network

Following function define the architecture of the model. Model consists of one dimensional convolution layers and LSTM layers to handle longer sequences. The architecture definition needs to be fed into FEModel along with model name and optimizer.


```python
def create_lstm():
    model = tf.keras.Sequential()
    model.add(layers.Embedding(MAX_WORDS, 64, input_length=MAX_LEN))
    model.add(layers.Conv1D(32, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=4))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(250, activation='relu'))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model
```

Next, fe.Network takes series of operators and here we feed our FEModel in the ModelOp with inputs and outputs. It should be noted that the <i>y_pred</i> is the key in the data dictionary which will store the predictions.


```python
model = fe.build(model_def=create_lstm, model_name='lstm_imdb', optimizer='adam', loss_name="loss")
#define the network
network = fe.Network(ops=[
        ModelOp(inputs="x", model=model, outputs="y_pred"),
                        BinaryCrossentropy(y_true="y", y_pred="y_pred", outputs="loss")
    ])
```

## Step 3: Prepare estimator and configure the training loop

In the training loop, we want to measure the validation loss and save the model that has the minimum loss. ModelSaver and Accuracy in the Trace class provide this convenient feature of storing the model.


```python
save_dir = tempfile.mkdtemp()
traces = [Accuracy(true_key="y", pred_key="y_pred", output_name='acc'),
         ModelSaver(model_name="lstm_imdb", save_dir=save_dir, save_best=True)]
```

We can define the estimator specifying the traning configurations and fit the model.


```python
estimator = fe.Estimator(network=network, 
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps,
                         pipeline=pipeline, 
                         epochs=epochs,
                         traces=traces)
estimator.fit()
```

## Inferencing

The best model is stored in temporary directory. We can load the model and perform the inference on sampled sequence from evaluation set.


```python
model_name = 'lstm_imdb_best_loss.h5'
model_path = os.path.join(save_dir, model_name)
trained_model = tf.keras.models.load_model(model_path, compile=False)
```

Get any random sequence and compare the prediction with the ground truth.


```python
selected_idx = np.random.randint(10000)
print("Ground truth is: ",y_eval[selected_idx])
padded_seq = np.array([pad(x_eval[selected_idx], MAX_LEN, 0)])
prediction = trained_model.predict(padded_seq)
print("Prediction for the input sequence: ",prediction)
```

    Ground truth is:  0
    Prediction for the input sequence:  [[0.01495596]]

