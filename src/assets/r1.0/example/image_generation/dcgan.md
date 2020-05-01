<h1>DCGAN Example with MNIST Dataset</h1>


```python
import tempfile
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from matplotlib import pyplot as plt
import fastestimator as fe
from fastestimator.backend import binary_crossentropy, feed_forward
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.numpyop.univariate import ExpandDims, Normalize
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import ModelSaver
```


```python
batch_size = 256
epochs = 50
max_steps_per_epoch = None
save_dir = tempfile.mkdtemp()
model_name = 'model_epoch_50.h5'
```

<h2>Building components</h2>

<h3>Step 1: Prepare training and define pipeline</h3>

We are loading data from tf.keras.datasets.mnist and defining series of operations to perform on the data before the training


```python
train_data, _ = mnist.load_data()
pipeline = fe.Pipeline(
    train_data=train_data,
    batch_size=batch_size,
    ops=[
        ExpandDims(inputs="x", outputs="x"),
        Normalize(inputs="x", outputs="x", mean=1.0, std=1.0, max_pixel_value=127.5),
        NumpyOp(inputs=lambda: np.random.normal(size=[100]).astype('float32'), outputs="z")
    ])
```

<h3>Step 2: Create model and FastEstimator network</h3>

First, we have to define the network architecture for both <b>Generator</b> and <b>Discriminator</b>. After defining the architecture, users are expected to feed the architecture definition, its associated model name and optimizer to fe.build.


```python
def generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100, )))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model
```


```python
def discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model
```


```python
gen_model = fe.build(model_fn=generator, optimizer_fn=lambda: tf.optimizers.Adam(1e-4))
disc_model = fe.build(model_fn=discriminator, optimizer_fn=lambda: tf.optimizers.Adam(1e-4))
```

We define the generator and discriminator loss. Loss can have multiple inputs and outputs.


```python
class GLoss(TensorOp):
    """Compute generator loss."""
    def forward(self, data, state):
        return binary_crossentropy(y_pred=data, y_true=tf.ones_like(data), from_logits=True)
```


```python
class DLoss(TensorOp):
    """Compute discriminator loss."""
    def forward(self, data, state):
        true_score, fake_score = data
        real_loss = binary_crossentropy(y_pred=true_score, y_true=tf.ones_like(true_score), from_logits=True)
        fake_loss = binary_crossentropy(y_pred=fake_score, y_true=tf.zeros_like(fake_score), from_logits=True)
        total_loss = real_loss + fake_loss
        return total_loss
```

Here, <i>fe.Network</i> takes series of operators and here we feed our model in the ModelOp with inputs and outputs. Also, group our model with the loss functions that we defined earlier


```python
network = fe.Network(ops=[
        ModelOp(model=gen_model, inputs="z", outputs="x_fake"),
        ModelOp(model=disc_model, inputs="x_fake", outputs="fake_score"),
        GLoss(inputs="fake_score", outputs="gloss"),
        UpdateOp(model=gen_model, loss_name="gloss"),
        ModelOp(inputs="x", model=disc_model, outputs="true_score"),
        DLoss(inputs=("true_score", "fake_score"), outputs="dloss"),
        UpdateOp(model=disc_model, loss_name="dloss")
    ])
```

<h3>Step 3: Prepare estimator and configure the training loop</h3>

We will define Estimator that has four arguments network, pipeline, epochs and traces. Network and Pipeline objects are passed here as an argument along with number of epochs and traces.

We will define traces to save the model with frequency of five that mean it will save the model at every 5 epochs.


```python
traces=ModelSaver(model=gen_model, save_dir=save_dir, frequency=5)
```


```python
estimator = fe.Estimator(pipeline=pipeline,
                         network=network,
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
                                                                            
    
    FastEstimator-Start: step: 1; model_lr: 1e-04; model1_lr: 1e-04; 
    FastEstimator-Train: step: 1; gloss: 0.7122225; dloss: 1.3922014; 
    FastEstimator-Train: step: 100; gloss: 0.906471; dloss: 0.8187004; steps/sec: 10.09; 
    FastEstimator-Train: step: 200; gloss: 0.59155834; dloss: 1.5896755; steps/sec: 9.93; 
    FastEstimator-Train: step: 235; epoch: 1; epoch_time: 28.53 sec; 
    FastEstimator-Train: step: 300; gloss: 0.7163421; dloss: 1.3333399; steps/sec: 8.9; 
    FastEstimator-Train: step: 400; gloss: 0.6816584; dloss: 1.6007018; steps/sec: 9.88; 
    FastEstimator-Train: step: 470; epoch: 2; epoch_time: 23.95 sec; 
    FastEstimator-Train: step: 500; gloss: 0.7051203; dloss: 1.4395489; steps/sec: 9.69; 
    FastEstimator-Train: step: 600; gloss: 0.75529504; dloss: 1.2358603; steps/sec: 9.86; 
    FastEstimator-Train: step: 700; gloss: 0.8082159; dloss: 1.2728964; steps/sec: 9.84; 
    FastEstimator-Train: step: 705; epoch: 3; epoch_time: 24.03 sec; 
    FastEstimator-Train: step: 800; gloss: 0.8434949; dloss: 1.3006642; steps/sec: 9.65; 
    FastEstimator-Train: step: 900; gloss: 0.84470236; dloss: 1.2344811; steps/sec: 9.79; 
    FastEstimator-Train: step: 940; epoch: 4; epoch_time: 24.16 sec; 
    FastEstimator-Train: step: 1000; gloss: 0.9431131; dloss: 1.0444374; steps/sec: 9.66; 
    FastEstimator-Train: step: 1100; gloss: 0.6982814; dloss: 1.5213135; steps/sec: 9.77; 
    Saved model to /tmp/tmpspul4xo8/model_epoch_5.h5
    FastEstimator-Train: step: 1175; epoch: 5; epoch_time: 24.17 sec; 
    FastEstimator-Train: step: 1200; gloss: 1.2540445; dloss: 0.8278078; steps/sec: 9.63; 
    FastEstimator-Train: step: 1300; gloss: 0.70465124; dloss: 1.7595482; steps/sec: 9.76; 
    FastEstimator-Train: step: 1400; gloss: 0.83103234; dloss: 1.168882; steps/sec: 9.77; 
    FastEstimator-Train: step: 1410; epoch: 6; epoch_time: 24.22 sec; 
    FastEstimator-Train: step: 1500; gloss: 0.86833733; dloss: 1.2078841; steps/sec: 9.57; 
    FastEstimator-Train: step: 1600; gloss: 0.82795817; dloss: 1.2242851; steps/sec: 9.75; 
    FastEstimator-Train: step: 1645; epoch: 7; epoch_time: 24.3 sec; 
    FastEstimator-Train: step: 1700; gloss: 0.9743507; dloss: 1.0731742; steps/sec: 9.59; 
    FastEstimator-Train: step: 1800; gloss: 0.89325964; dloss: 1.1766281; steps/sec: 9.76; 
    FastEstimator-Train: step: 1880; epoch: 8; epoch_time: 24.25 sec; 
    FastEstimator-Train: step: 1900; gloss: 1.0287898; dloss: 0.9916363; steps/sec: 9.6; 
    FastEstimator-Train: step: 2000; gloss: 0.8240694; dloss: 1.313368; steps/sec: 9.74; 
    FastEstimator-Train: step: 2100; gloss: 0.9738071; dloss: 1.1259043; steps/sec: 9.73; 
    FastEstimator-Train: step: 2115; epoch: 9; epoch_time: 24.3 sec; 
    FastEstimator-Train: step: 2200; gloss: 1.0899432; dloss: 1.0272337; steps/sec: 9.57; 
    FastEstimator-Train: step: 2300; gloss: 0.868231; dloss: 1.2400149; steps/sec: 9.72; 
    Saved model to /tmp/tmpspul4xo8/model_epoch_10.h5
    FastEstimator-Train: step: 2350; epoch: 10; epoch_time: 24.35 sec; 
    FastEstimator-Train: step: 2400; gloss: 0.9001913; dloss: 1.1931081; steps/sec: 9.58; 
    FastEstimator-Train: step: 2500; gloss: 1.0865673; dloss: 0.8990781; steps/sec: 9.71; 
    FastEstimator-Train: step: 2585; epoch: 11; epoch_time: 24.34 sec; 
    FastEstimator-Train: step: 2600; gloss: 0.7485407; dloss: 1.672249; steps/sec: 9.58; 
    FastEstimator-Train: step: 2700; gloss: 1.045316; dloss: 1.0383615; steps/sec: 9.73; 
    FastEstimator-Train: step: 2800; gloss: 0.7666995; dloss: 1.4343789; steps/sec: 9.72; 
    FastEstimator-Train: step: 2820; epoch: 12; epoch_time: 24.3 sec; 
    FastEstimator-Train: step: 2900; gloss: 1.1756387; dloss: 0.96622103; steps/sec: 9.6; 
    FastEstimator-Train: step: 3000; gloss: 0.9090629; dloss: 1.1984154; steps/sec: 9.72; 
    FastEstimator-Train: step: 3055; epoch: 13; epoch_time: 24.29 sec; 
    FastEstimator-Train: step: 3100; gloss: 0.9301505; dloss: 1.113826; steps/sec: 9.59; 
    FastEstimator-Train: step: 3200; gloss: 0.99965835; dloss: 1.0707076; steps/sec: 9.74; 
    FastEstimator-Train: step: 3290; epoch: 14; epoch_time: 24.31 sec; 
    FastEstimator-Train: step: 3300; gloss: 0.80838567; dloss: 1.6384692; steps/sec: 9.55; 
    FastEstimator-Train: step: 3400; gloss: 0.8714433; dloss: 1.326818; steps/sec: 9.74; 
    FastEstimator-Train: step: 3500; gloss: 0.9549879; dloss: 1.2086997; steps/sec: 9.73; 
    Saved model to /tmp/tmpspul4xo8/model_epoch_15.h5
    FastEstimator-Train: step: 3525; epoch: 15; epoch_time: 24.34 sec; 
    FastEstimator-Train: step: 3600; gloss: 1.0164418; dloss: 1.0690243; steps/sec: 9.57; 
    FastEstimator-Train: step: 3700; gloss: 1.0357686; dloss: 1.0537144; steps/sec: 9.72; 
    FastEstimator-Train: step: 3760; epoch: 16; epoch_time: 24.32 sec; 
    FastEstimator-Train: step: 3800; gloss: 0.7402923; dloss: 1.4840925; steps/sec: 9.6; 
    FastEstimator-Train: step: 3900; gloss: 0.91192436; dloss: 1.3617609; steps/sec: 9.72; 
    FastEstimator-Train: step: 3995; epoch: 17; epoch_time: 24.3 sec; 
    FastEstimator-Train: step: 4000; gloss: 1.2626994; dloss: 0.9568275; steps/sec: 9.59; 
    FastEstimator-Train: step: 4100; gloss: 0.97824305; dloss: 1.300906; steps/sec: 9.74; 
    FastEstimator-Train: step: 4200; gloss: 0.93075603; dloss: 1.387594; steps/sec: 9.73; 
    FastEstimator-Train: step: 4230; epoch: 18; epoch_time: 24.31 sec; 
    FastEstimator-Train: step: 4300; gloss: 1.0180345; dloss: 1.0898602; steps/sec: 9.59; 
    FastEstimator-Train: step: 4400; gloss: 1.051662; dloss: 1.3392837; steps/sec: 9.71; 
    FastEstimator-Train: step: 4465; epoch: 19; epoch_time: 24.33 sec; 
    FastEstimator-Train: step: 4500; gloss: 1.0151768; dloss: 1.1482071; steps/sec: 9.56; 
    FastEstimator-Train: step: 4600; gloss: 1.107022; dloss: 0.96815336; steps/sec: 9.71; 
    FastEstimator-Train: step: 4700; gloss: 1.0924942; dloss: 1.1389236; steps/sec: 9.72; 
    Saved model to /tmp/tmpspul4xo8/model_epoch_20.h5
    FastEstimator-Train: step: 4700; epoch: 20; epoch_time: 24.39 sec; 
    FastEstimator-Train: step: 4800; gloss: 1.1345683; dloss: 1.2068424; steps/sec: 9.54; 
    FastEstimator-Train: step: 4900; gloss: 1.142304; dloss: 0.9673606; steps/sec: 9.74; 
    FastEstimator-Train: step: 4935; epoch: 21; epoch_time: 24.3 sec; 
    FastEstimator-Train: step: 5000; gloss: 0.9886; dloss: 1.0960109; steps/sec: 9.57; 
    FastEstimator-Train: step: 5100; gloss: 0.8936993; dloss: 1.2922779; steps/sec: 9.74; 
    FastEstimator-Train: step: 5170; epoch: 22; epoch_time: 24.3 sec; 
    FastEstimator-Train: step: 5200; gloss: 1.1095095; dloss: 1.1243165; steps/sec: 9.57; 
    FastEstimator-Train: step: 5300; gloss: 1.2485275; dloss: 0.89292765; steps/sec: 9.74; 
    FastEstimator-Train: step: 5400; gloss: 1.0476826; dloss: 1.2994311; steps/sec: 9.75; 
    FastEstimator-Train: step: 5405; epoch: 23; epoch_time: 24.29 sec; 
    FastEstimator-Train: step: 5500; gloss: 1.3308836; dloss: 0.871735; steps/sec: 9.63; 
    FastEstimator-Train: step: 5600; gloss: 1.115385; dloss: 1.2837725; steps/sec: 9.74; 
    FastEstimator-Train: step: 5640; epoch: 24; epoch_time: 24.23 sec; 
    FastEstimator-Train: step: 5700; gloss: 1.1920481; dloss: 1.0993654; steps/sec: 9.62; 
    FastEstimator-Train: step: 5800; gloss: 1.3005233; dloss: 0.914361; steps/sec: 9.74; 
    Saved model to /tmp/tmpspul4xo8/model_epoch_25.h5
    FastEstimator-Train: step: 5875; epoch: 25; epoch_time: 24.27 sec; 
    FastEstimator-Train: step: 5900; gloss: 1.3146336; dloss: 0.8816396; steps/sec: 9.6; 
    FastEstimator-Train: step: 6000; gloss: 0.9764897; dloss: 1.289681; steps/sec: 9.75; 
    FastEstimator-Train: step: 6100; gloss: 1.1467731; dloss: 1.1918977; steps/sec: 9.75; 
    FastEstimator-Train: step: 6110; epoch: 26; epoch_time: 24.26 sec; 
    FastEstimator-Train: step: 6200; gloss: 1.6301311; dloss: 0.9541445; steps/sec: 9.6; 
    FastEstimator-Train: step: 6300; gloss: 1.2840165; dloss: 0.9587291; steps/sec: 9.73; 
    FastEstimator-Train: step: 6345; epoch: 27; epoch_time: 24.3 sec; 
    FastEstimator-Train: step: 6400; gloss: 1.1097628; dloss: 1.0090048; steps/sec: 9.59; 
    FastEstimator-Train: step: 6500; gloss: 1.2495477; dloss: 0.89897555; steps/sec: 9.73; 
    FastEstimator-Train: step: 6580; epoch: 28; epoch_time: 24.32 sec; 
    FastEstimator-Train: step: 6600; gloss: 1.1773547; dloss: 1.1330662; steps/sec: 9.58; 
    FastEstimator-Train: step: 6700; gloss: 1.246088; dloss: 0.8964198; steps/sec: 9.73; 
    FastEstimator-Train: step: 6800; gloss: 1.2250234; dloss: 1.0358574; steps/sec: 9.73; 
    FastEstimator-Train: step: 6815; epoch: 29; epoch_time: 24.3 sec; 
    FastEstimator-Train: step: 6900; gloss: 1.1256618; dloss: 1.196687; steps/sec: 9.59; 
    FastEstimator-Train: step: 7000; gloss: 1.1131527; dloss: 1.0596428; steps/sec: 9.73; 
    Saved model to /tmp/tmpspul4xo8/model_epoch_30.h5
    FastEstimator-Train: step: 7050; epoch: 30; epoch_time: 24.31 sec; 
    FastEstimator-Train: step: 7100; gloss: 1.1662202; dloss: 1.0555116; steps/sec: 9.57; 
    FastEstimator-Train: step: 7200; gloss: 1.0653521; dloss: 1.1444951; steps/sec: 9.71; 
    FastEstimator-Train: step: 7285; epoch: 31; epoch_time: 24.36 sec; 
    FastEstimator-Train: step: 7300; gloss: 1.1732882; dloss: 1.1456137; steps/sec: 9.56; 
    FastEstimator-Train: step: 7400; gloss: 1.0872216; dloss: 1.128233; steps/sec: 9.74; 
    FastEstimator-Train: step: 7500; gloss: 1.2431256; dloss: 1.1538315; steps/sec: 9.73; 
    FastEstimator-Train: step: 7520; epoch: 32; epoch_time: 24.31 sec; 
    FastEstimator-Train: step: 7600; gloss: 1.0806718; dloss: 1.2206206; steps/sec: 9.57; 
    FastEstimator-Train: step: 7700; gloss: 1.1804712; dloss: 1.1420157; steps/sec: 9.71; 
    FastEstimator-Train: step: 7755; epoch: 33; epoch_time: 24.36 sec; 
    FastEstimator-Train: step: 7800; gloss: 1.1762993; dloss: 1.0413929; steps/sec: 9.54; 
    FastEstimator-Train: step: 7900; gloss: 1.2267275; dloss: 0.98290396; steps/sec: 9.71; 
    FastEstimator-Train: step: 7990; epoch: 34; epoch_time: 24.37 sec; 
    FastEstimator-Train: step: 8000; gloss: 1.1847881; dloss: 1.0905983; steps/sec: 9.55; 
    FastEstimator-Train: step: 8100; gloss: 1.1490288; dloss: 1.1739209; steps/sec: 9.72; 
    FastEstimator-Train: step: 8200; gloss: 1.0283768; dloss: 1.059457; steps/sec: 9.73; 
    Saved model to /tmp/tmpspul4xo8/model_epoch_35.h5
    FastEstimator-Train: step: 8225; epoch: 35; epoch_time: 24.33 sec; 
    FastEstimator-Train: step: 8300; gloss: 1.2351133; dloss: 1.1085691; steps/sec: 9.59; 
    FastEstimator-Train: step: 8400; gloss: 1.1488228; dloss: 1.1410246; steps/sec: 9.74; 
    FastEstimator-Train: step: 8460; epoch: 36; epoch_time: 24.31 sec; 
    FastEstimator-Train: step: 8500; gloss: 1.152446; dloss: 1.1371456; steps/sec: 9.55; 
    FastEstimator-Train: step: 8600; gloss: 1.2175394; dloss: 1.1543391; steps/sec: 9.74; 
    FastEstimator-Train: step: 8695; epoch: 37; epoch_time: 24.31 sec; 
    FastEstimator-Train: step: 8700; gloss: 1.1803217; dloss: 1.1517241; steps/sec: 9.59; 
    FastEstimator-Train: step: 8800; gloss: 0.9561673; dloss: 1.2418871; steps/sec: 9.77; 
    FastEstimator-Train: step: 8900; gloss: 1.0239995; dloss: 1.243228; steps/sec: 9.74; 
    FastEstimator-Train: step: 8930; epoch: 38; epoch_time: 24.26 sec; 
    FastEstimator-Train: step: 9000; gloss: 0.98074543; dloss: 1.2558163; steps/sec: 9.59; 
    FastEstimator-Train: step: 9100; gloss: 1.0084043; dloss: 1.2773612; steps/sec: 9.74; 
    FastEstimator-Train: step: 9165; epoch: 39; epoch_time: 24.3 sec; 
    FastEstimator-Train: step: 9200; gloss: 1.0313301; dloss: 1.2312038; steps/sec: 9.58; 
    FastEstimator-Train: step: 9300; gloss: 1.0100834; dloss: 1.2482088; steps/sec: 9.73; 
    FastEstimator-Train: step: 9400; gloss: 0.9327201; dloss: 1.3188391; steps/sec: 9.73; 
    Saved model to /tmp/tmpspul4xo8/model_epoch_40.h5
    FastEstimator-Train: step: 9400; epoch: 40; epoch_time: 24.34 sec; 
    FastEstimator-Train: step: 9500; gloss: 1.1315899; dloss: 1.1447232; steps/sec: 9.55; 
    FastEstimator-Train: step: 9600; gloss: 1.1352619; dloss: 1.0802212; steps/sec: 9.72; 
    FastEstimator-Train: step: 9635; epoch: 41; epoch_time: 24.3 sec; 
    FastEstimator-Train: step: 9700; gloss: 1.01453; dloss: 1.0975459; steps/sec: 9.59; 
    FastEstimator-Train: step: 9800; gloss: 0.90930146; dloss: 1.2942967; steps/sec: 9.76; 
    FastEstimator-Train: step: 9870; epoch: 42; epoch_time: 24.27 sec; 
    FastEstimator-Train: step: 9900; gloss: 1.0540565; dloss: 1.170531; steps/sec: 9.58; 
    FastEstimator-Train: step: 10000; gloss: 1.061863; dloss: 1.2722391; steps/sec: 9.76; 
    FastEstimator-Train: step: 10100; gloss: 0.9647354; dloss: 1.1689386; steps/sec: 9.73; 
    FastEstimator-Train: step: 10105; epoch: 43; epoch_time: 24.29 sec; 
    FastEstimator-Train: step: 10200; gloss: 1.2080085; dloss: 1.075758; steps/sec: 9.57; 
    FastEstimator-Train: step: 10300; gloss: 1.0741084; dloss: 1.1352613; steps/sec: 9.72; 
    FastEstimator-Train: step: 10340; epoch: 44; epoch_time: 24.34 sec; 
    FastEstimator-Train: step: 10400; gloss: 1.1394867; dloss: 0.9788251; steps/sec: 9.58; 
    FastEstimator-Train: step: 10500; gloss: 1.1983887; dloss: 1.0792823; steps/sec: 9.73; 
    Saved model to /tmp/tmpspul4xo8/model_epoch_45.h5
    FastEstimator-Train: step: 10575; epoch: 45; epoch_time: 24.31 sec; 
    FastEstimator-Train: step: 10600; gloss: 0.96989757; dloss: 1.2448618; steps/sec: 9.59; 
    FastEstimator-Train: step: 10700; gloss: 0.9579427; dloss: 1.2773428; steps/sec: 9.7; 
    FastEstimator-Train: step: 10800; gloss: 0.97453225; dloss: 1.2194138; steps/sec: 9.72; 
    FastEstimator-Train: step: 10810; epoch: 46; epoch_time: 24.36 sec; 
    FastEstimator-Train: step: 10900; gloss: 1.0218571; dloss: 1.1765122; steps/sec: 9.57; 
    FastEstimator-Train: step: 11000; gloss: 1.1221988; dloss: 1.1675267; steps/sec: 9.71; 
    FastEstimator-Train: step: 11045; epoch: 47; epoch_time: 24.34 sec; 
    FastEstimator-Train: step: 11100; gloss: 0.900293; dloss: 1.1741953; steps/sec: 9.57; 
    FastEstimator-Train: step: 11200; gloss: 1.1080045; dloss: 1.1090837; steps/sec: 9.73; 
    FastEstimator-Train: step: 11280; epoch: 48; epoch_time: 24.31 sec; 
    FastEstimator-Train: step: 11300; gloss: 1.1028197; dloss: 1.1044464; steps/sec: 9.59; 
    FastEstimator-Train: step: 11400; gloss: 1.0530615; dloss: 1.2150866; steps/sec: 9.74; 
    FastEstimator-Train: step: 11500; gloss: 0.8997061; dloss: 1.3101699; steps/sec: 9.75; 
    FastEstimator-Train: step: 11515; epoch: 49; epoch_time: 24.29 sec; 
    FastEstimator-Train: step: 11600; gloss: 1.0087067; dloss: 1.3297874; steps/sec: 9.58; 
    FastEstimator-Train: step: 11700; gloss: 1.0053492; dloss: 1.2499433; steps/sec: 9.74; 
    Saved model to /tmp/tmpspul4xo8/model_epoch_50.h5
    FastEstimator-Train: step: 11750; epoch: 50; epoch_time: 24.3 sec; 
    FastEstimator-Finish: step: 11750; total_time: 1219.47 sec; model_lr: 1e-04; model1_lr: 1e-04; 


<h2>Inferencing</h2>

For inferencing, first we have to load the trained model weights. We will load the trained generator weights using <i>fe.build</i>


```python
model_path = os.path.join(save_dir, model_name)
trained_model = fe.build(model_fn=generator, weights_path=model_path, optimizer_fn=lambda: tf.optimizers.Adam(1e-4))
```

    Loaded model weights from /tmp/tmpspul4xo8/model_epoch_50.h5


We will generate the images from the random noise.


```python
images = feed_forward(trained_model, np.random.normal(size=(16, 100)), training=False)
```

    WARNING:tensorflow:Layer dense_4 is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
    If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
    To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    



```python
fig, axes = plt.subplots(4, 4)
axes = np.ravel(axes)
for i in range(images.shape[0]):
    axes[i].axis('off')
    axes[i].imshow(np.squeeze(images[i, ...] * 127.5 + 127.5), cmap='gray')
```


![png](assets/example/image_generation/dcgan_files/dcgan_29_0.png)



```python

```
