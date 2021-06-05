# MNIST Image Classification Using LeNet (Tensorflow Backend)
In this example, we are going to demonstrate how to train an MNIST image classification model using a LeNet model architecture and TensorFlow backend. 

## Import the required libraries


```python
import tensorflow as tf
import fastestimator as fe
import numpy as np
import matplotlib.pyplot as plt
import tempfile
```


```python
#training parameters
epochs = 2
batch_size = 32
max_train_steps_per_epoch = None
max_eval_steps_per_epoch = None
save_dir = tempfile.mkdtemp()
```

## Step 1 - Data and `Pipeline` preparation
In this step, we will load MNIST training and validation datasets and prepare FastEstimator's pipeline.

### Load dataset 
We use a FastEstimator API to load the MNIST dataset and then get a test set by splitting 50% of the data off of the evaluation set. 


```python
from fastestimator.dataset.data import mnist

train_data, eval_data = mnist.load_data()
test_data = eval_data.split(0.5)
```

### Set up a preprocessing pipeline
In this example, the data preprocessing steps include adding a channel to the images (since they are grey-scale) and normalizing the image pixel values to the range [0, 1]. We set up these processing steps using `Ops`. The `Pipeline` also takes our data sources and batch size as inputs. 


```python
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax

pipeline = fe.Pipeline(train_data=train_data,
                       eval_data=eval_data,
                       test_data=test_data,
                       batch_size=batch_size,
                       ops=[ExpandDims(inputs="x", outputs="x_out"), 
                            Minmax(inputs="x_out", outputs="x_out")])
```

### Validate `Pipeline`
In order to make sure the pipeline works as expected, we need to visualize its output. `Pipeline.get_results` will return a batch  of pipeline output to enable this:  


```python
data = pipeline.get_results()
data_xin = data["x"]
data_xout = data["x_out"]

print("the pipeline input data size: {}".format(data_xin.numpy().shape))
print("the pipeline output data size: {}".format(data_xout.numpy().shape))
print("the maximum pixel value of output image: {}".format(np.max(data_xout.numpy())))
print("the minimum pixel value of output image: {}".format(np.min(data_xout.numpy())))
```

    the pipeline input data size: (32, 28, 28)
    the pipeline output data size: (32, 28, 28, 1)
    the maximum pixel value of output image: 1.0
    the minimum pixel value of output image: 0.0



```python
num_samples = 5
indices = np.random.choice(batch_size, size=num_samples, replace=False)
inputs = tf.gather(data_xin.numpy(), indices)
outputs = tf.gather(data_xout.numpy(), indices)
img = fe.util.ImgData(pipeline_input=inputs, pipeline_output=outputs)
fig = img.paint_figure()
```


    
![png](./assets/branches/r1.2/example/image_classification/mnist/mnist_files/mnist_10_0.png)
    


## Step 2 - `Network` construction
**FastEstimator supports both PyTorch and TensorFlow, so this section could use either backend.** <br>
We are going to only demonstrate the TensorFlow backend in this example.

### Model construction
Here we are going to import one of FastEstimator's pre-defined model architectures, which was written in TensorFlow. We create a model instance by compiling our model definition function along with a specific model optimizer.


```python
from fastestimator.architecture.tensorflow import LeNet

model = fe.build(model_fn=LeNet, optimizer_fn="adam")
```

### `Network` definition
We are going to connect the model and `Ops` together into a `Network`. `Ops` are the basic components of a `Network`. They can be logic for loss calculation, model update rules, or even models themselves. 


```python
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp


network = fe.Network(ops=[
        ModelOp(model=model, inputs="x_out", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
```

## Step 3 - `Estimator` definition and training
In this step, we define an `Estimator` to connect our `Network` with our `Pipeline` and set the `traces` which compute accuracy (`Accuracy`), save the best model (`BestModelSaver`), and change the model learning rate over time (`LRScheduler`).


```python
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


traces = [
    Accuracy(true_key="y", pred_key="y_pred"),
    BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max"),
    LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=3750, init_lr=1e-3))
]

estimator = fe.Estimator(pipeline=pipeline,
                         network=network,
                         epochs=epochs,
                         traces=traces,
                         max_train_steps_per_epoch=max_train_steps_per_epoch,
                         max_eval_steps_per_epoch=max_eval_steps_per_epoch)

estimator.fit() # start the training process
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 100; 
    FastEstimator-Train: step: 1; ce: 2.3120923; model_lr: 0.001; 
    FastEstimator-Train: step: 100; ce: 0.3962714; steps/sec: 130.51; model_lr: 0.000998283; 
    FastEstimator-Train: step: 200; ce: 0.1641247; steps/sec: 128.14; model_lr: 0.0009930746; 
    FastEstimator-Train: step: 300; ce: 0.20016384; steps/sec: 123.16; model_lr: 0.0009844112; 
    FastEstimator-Train: step: 400; ce: 0.139256; steps/sec: 118.37; model_lr: 0.00097235345; 
    FastEstimator-Train: step: 500; ce: 0.20949002; steps/sec: 116.86; model_lr: 0.000956986; 
    FastEstimator-Train: step: 600; ce: 0.08091536; steps/sec: 115.89; model_lr: 0.00093841663; 
    FastEstimator-Train: step: 700; ce: 0.069529384; steps/sec: 112.6; model_lr: 0.0009167756; 
    FastEstimator-Train: step: 800; ce: 0.02633699; steps/sec: 109.37; model_lr: 0.00089221465; 
    FastEstimator-Train: step: 900; ce: 0.12905718; steps/sec: 104.44; model_lr: 0.0008649062; 
    FastEstimator-Train: step: 1000; ce: 0.018508099; steps/sec: 108.77; model_lr: 0.0008350416; 
    FastEstimator-Train: step: 1100; ce: 0.10962237; steps/sec: 106.61; model_lr: 0.00080283044; 
    FastEstimator-Train: step: 1200; ce: 0.047606118; steps/sec: 101.79; model_lr: 0.0007684987; 
    FastEstimator-Train: step: 1300; ce: 0.13268313; steps/sec: 97.41; model_lr: 0.0007322871; 
    FastEstimator-Train: step: 1400; ce: 0.026097888; steps/sec: 94.08; model_lr: 0.00069444976; 
    FastEstimator-Train: step: 1500; ce: 0.020507228; steps/sec: 92.03; model_lr: 0.0006552519; 
    FastEstimator-Train: step: 1600; ce: 0.0048278654; steps/sec: 92.14; model_lr: 0.00061496865; 
    FastEstimator-Train: step: 1700; ce: 0.01370596; steps/sec: 88.92; model_lr: 0.0005738824; 
    FastEstimator-Train: step: 1800; ce: 0.15647383; steps/sec: 89.46; model_lr: 0.00053228147; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 19.9 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp_19zst_o/model_best_accuracy
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.048851434; accuracy: 0.985; since_best_accuracy: 0; max_accuracy: 0.985; 
    FastEstimator-Train: step: 1900; ce: 0.0039410545; steps/sec: 87.28; model_lr: 0.00049045763; 
    FastEstimator-Train: step: 2000; ce: 0.017347965; steps/sec: 86.28; model_lr: 0.00044870423; 
    FastEstimator-Train: step: 2100; ce: 0.0798438; steps/sec: 89.34; model_lr: 0.0004073141; 
    FastEstimator-Train: step: 2200; ce: 0.17821585; steps/sec: 86.08; model_lr: 0.00036657765; 
    FastEstimator-Train: step: 2300; ce: 0.002756747; steps/sec: 86.16; model_lr: 0.00032678054; 
    FastEstimator-Train: step: 2400; ce: 0.00071113696; steps/sec: 85.27; model_lr: 0.00028820196; 
    FastEstimator-Train: step: 2500; ce: 0.0027370513; steps/sec: 80.7; model_lr: 0.00025111248; 
    FastEstimator-Train: step: 2600; ce: 0.0123588; steps/sec: 84.42; model_lr: 0.00021577229; 
    FastEstimator-Train: step: 2700; ce: 0.0069204723; steps/sec: 81.53; model_lr: 0.00018242926; 
    FastEstimator-Train: step: 2800; ce: 0.00642678; steps/sec: 77.9; model_lr: 0.00015131726; 
    FastEstimator-Train: step: 2900; ce: 0.008096467; steps/sec: 81.94; model_lr: 0.00012265453; 
    FastEstimator-Train: step: 3000; ce: 0.0023379987; steps/sec: 79.61; model_lr: 9.664212e-05; 
    FastEstimator-Train: step: 3100; ce: 0.104631886; steps/sec: 79.75; model_lr: 7.346248e-05; 
    FastEstimator-Train: step: 3200; ce: 0.003903159; steps/sec: 78.32; model_lr: 5.3278196e-05; 
    FastEstimator-Train: step: 3300; ce: 0.024989499; steps/sec: 79.65; model_lr: 3.6230853e-05; 
    FastEstimator-Train: step: 3400; ce: 0.02124865; steps/sec: 81.16; model_lr: 2.2440026e-05; 
    FastEstimator-Train: step: 3500; ce: 0.008722213; steps/sec: 77.56; model_lr: 1.2002448e-05; 
    FastEstimator-Train: step: 3600; ce: 0.18177237; steps/sec: 77.0; model_lr: 4.9913274e-06; 
    FastEstimator-Train: step: 3700; ce: 0.005134264; steps/sec: 79.63; model_lr: 1.4558448e-06; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 22.99 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmp_19zst_o/model_best_accuracy
    FastEstimator-Eval: step: 3750; epoch: 2; ce: 0.021919494; accuracy: 0.994; since_best_accuracy: 0; max_accuracy: 0.994; 
    FastEstimator-Finish: step: 3750; total_time: 46.88 sec; model_lr: 1.0001753e-06; 


## Model testing
`Estimator.test` triggers model testing using the test dataset that was specified in `Pipeline`. We can evaluate the model's accuracy on this previously unseen data. 


```python
estimator.test()
```

    FastEstimator-Test: step: 3750; epoch: 2; accuracy: 0.9908; 


## Inferencing
Now let's run inferencing on several images directly using the model that we just trained. 
We randomly select 5 images from the testing dataset and infer them image by image by leveraging `Pipeline.transform` and `Network.transform`:


```python
num_samples = 5
indices = np.random.choice(batch_size, size=num_samples, replace=False)

inputs = []
outputs = []
predictions = []

for idx in indices:
    inputs.append(test_data["x"][idx])
    data = {"x": inputs[-1]}
    
    # run the pipeline
    data = pipeline.transform(data, mode="infer") 
    outputs.append(data["x_out"].squeeze(axis=(0,3)))
    
    # run the network
    data = network.transform(data, mode="infer")
    predictions.append(np.argmax(data["y_pred"].numpy().squeeze(axis=(0))))

img = fe.util.ImgData(pipeline_input=np.stack(inputs), pipeline_output=np.stack(outputs), predictions=np.stack(predictions))
fig = img.paint_figure()
```


    
![png](./assets/branches/r1.2/example/image_classification/mnist/mnist_files/mnist_20_0.png)
    

