# MNIST Image Classification example using LeNet (Tensorflow backend)
In this example, we are going to demonstrate how to train a MNIST image classification model using LeNet model architeture with Tesnorflow backend. 

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
max_steps_per_epoch = None
save_dir = tempfile.mkdtemp()
```

## Step 1 - Data and `Pipeline` preparation
In this step, we will load MNIST training and validation datasets and prepare FastEstimator's pipeline.

### Load dataset 
We use fastestimator API to load the MNIST dataset and get the test set by splitting 50% evaluation set. 


```python
from fastestimator.dataset.data import mnist

train_data, eval_data = mnist.load_data()
test_data = eval_data.split(0.5)
```

### Set up preprocessing pipline
In this example, the data preprocessing steps include expanding image dimension and normalizing the pixel value to range [0, 1]. We set up those processing step using `Ops` and meanwhile define the data source (loaded dataset) and batch size. 


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
In order to make sure the pipeline works as expected, we need to visualize the output of pipeline image and check its size.
`Pipeline.get_results` will return a batch data of pipeline output. 
 


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
sample_num = 5

fig, axs = plt.subplots(sample_num, 2, figsize=(12,12))

axs[0,0].set_title("pipeline input")
axs[0,1].set_title("pipeline output")


for i, j in enumerate(np.random.randint(low=0, high=batch_size-1, size=sample_num)):
    img_in = data_xin.numpy()[j]
    axs[i,0].imshow(img_in, cmap="gray")
    
    img_out = data_xout.numpy()[j,:,:,0]
    axs[i,1].imshow(img_out, cmap="gray")
```


![png](assets/example/image_classification/mnist_files/mnist_10_0.png)


## Step 2 - `Network` construction
**FastEstimator supports both Pytorch and Tensorflow, so this section can use both backend to implement.** <br>
We are going to only demonstate the Tensorflow way in this example.

### Model construction
Here the model definition is going to be imported from the FastEstimator pre-defined architecture that is implemented in Tensorflow, and we create model instance by compiling it with specific model optimizer.


```python
from fastestimator.architecture.tensorflow import LeNet

model = fe.build(model_fn=LeNet, optimizer_fn="adam")
```

### `Network` definition
We are going to connect the model and `Ops` together into a `Network`. `Ops` are the basic component of `Network`. They can be logic for loss calculation, model update units, and even model itself is also considered as an `Op`. 


```python
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp


network = fe.Network(ops=[
        ModelOp(model=model, inputs="x_out", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce", mode="!infer")
    ])
```

## Step 3 - `Estimator` definition and training
In this step, we define the `Estimator` to connect the `Network` with `Pipeline` and set the `traces` which compute accuracy (Accuracy), save best model (BestModelSaver), and change learning rate (LRScheduler) 


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
                         max_steps_per_epoch=max_steps_per_epoch)

estimator.fit() # start the training process
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; model_lr: 0.001; 
    FastEstimator-Train: step: 1; ce: 2.2952478; model_lr: 0.001; 
    FastEstimator-Train: step: 100; ce: 0.3141091; steps/sec: 331.75; model_lr: 0.000998283; 
    FastEstimator-Train: step: 200; ce: 0.2921242; steps/sec: 327.33; model_lr: 0.0009930746; 
    FastEstimator-Train: step: 300; ce: 0.030078162; steps/sec: 339.11; model_lr: 0.0009844112; 
    FastEstimator-Train: step: 400; ce: 0.37753928; steps/sec: 339.46; model_lr: 0.00097235345; 
    FastEstimator-Train: step: 500; ce: 0.014425077; steps/sec: 328.28; model_lr: 0.000956986; 
    FastEstimator-Train: step: 600; ce: 0.061481424; steps/sec: 337.15; model_lr: 0.00093841663; 
    FastEstimator-Train: step: 700; ce: 0.18871143; steps/sec: 330.63; model_lr: 0.0009167756; 
    FastEstimator-Train: step: 800; ce: 0.04665528; steps/sec: 336.71; model_lr: 0.00089221465; 
    FastEstimator-Train: step: 900; ce: 0.16318482; steps/sec: 335.27; model_lr: 0.0008649062; 
    FastEstimator-Train: step: 1000; ce: 0.17174809; steps/sec: 339.01; model_lr: 0.0008350416; 
    FastEstimator-Train: step: 1100; ce: 0.072015464; steps/sec: 336.46; model_lr: 0.00080283044; 
    FastEstimator-Train: step: 1200; ce: 0.0038511527; steps/sec: 334.32; model_lr: 0.0007684987; 
    FastEstimator-Train: step: 1300; ce: 0.13038625; steps/sec: 338.65; model_lr: 0.0007322871; 
    FastEstimator-Train: step: 1400; ce: 0.001873189; steps/sec: 335.34; model_lr: 0.00069444976; 
    FastEstimator-Train: step: 1500; ce: 0.003812432; steps/sec: 341.54; model_lr: 0.0006552519; 
    FastEstimator-Train: step: 1600; ce: 0.118942365; steps/sec: 328.41; model_lr: 0.00061496865; 
    FastEstimator-Train: step: 1700; ce: 0.012451285; steps/sec: 335.49; model_lr: 0.0005738824; 
    FastEstimator-Train: step: 1800; ce: 0.082549796; steps/sec: 334.01; model_lr: 0.00053228147; 
    FastEstimator-Train: step: 1875; epoch: 1; epoch_time: 8.51 sec; 
    Saved model to /tmp/tmp6rwicno6/model_best_accuracy.h5
    FastEstimator-Eval: step: 1875; epoch: 1; ce: 0.044308867; min_ce: 0.044308867; since_best: 0; accuracy: 0.985; 
    FastEstimator-Train: step: 1900; ce: 0.046987697; steps/sec: 136.47; model_lr: 0.00049045763; 
    FastEstimator-Train: step: 2000; ce: 0.02985252; steps/sec: 331.95; model_lr: 0.00044870423; 
    FastEstimator-Train: step: 2100; ce: 0.06913616; steps/sec: 332.27; model_lr: 0.0004073141; 
    FastEstimator-Train: step: 2200; ce: 0.031003162; steps/sec: 339.06; model_lr: 0.00036657765; 
    FastEstimator-Train: step: 2300; ce: 0.020062737; steps/sec: 338.79; model_lr: 0.00032678054; 
    FastEstimator-Train: step: 2400; ce: 0.039522864; steps/sec: 341.69; model_lr: 0.00028820196; 
    FastEstimator-Train: step: 2500; ce: 0.041544776; steps/sec: 347.78; model_lr: 0.00025111248; 
    FastEstimator-Train: step: 2600; ce: 0.0073616905; steps/sec: 330.83; model_lr: 0.00021577229; 
    FastEstimator-Train: step: 2700; ce: 0.0032872492; steps/sec: 340.04; model_lr: 0.00018242926; 
    FastEstimator-Train: step: 2800; ce: 0.021862563; steps/sec: 333.89; model_lr: 0.00015131726; 
    FastEstimator-Train: step: 2900; ce: 0.10750027; steps/sec: 339.62; model_lr: 0.00012265453; 
    FastEstimator-Train: step: 3000; ce: 0.041685883; steps/sec: 336.79; model_lr: 9.664212e-05; 
    FastEstimator-Train: step: 3100; ce: 0.008866243; steps/sec: 339.07; model_lr: 7.346248e-05; 
    FastEstimator-Train: step: 3200; ce: 0.09189169; steps/sec: 338.88; model_lr: 5.3278196e-05; 
    FastEstimator-Train: step: 3300; ce: 0.08884057; steps/sec: 331.95; model_lr: 3.6230853e-05; 
    FastEstimator-Train: step: 3400; ce: 0.1302068; steps/sec: 342.97; model_lr: 2.2440026e-05; 
    FastEstimator-Train: step: 3500; ce: 0.112514116; steps/sec: 333.27; model_lr: 1.2002448e-05; 
    FastEstimator-Train: step: 3600; ce: 0.0024552555; steps/sec: 338.11; model_lr: 4.9913274e-06; 
    FastEstimator-Train: step: 3700; ce: 0.05180082; steps/sec: 334.38; model_lr: 1.4558448e-06; 
    FastEstimator-Train: step: 3750; epoch: 2; epoch_time: 6.0 sec; 
    Saved model to /tmp/tmp6rwicno6/model_best_accuracy.h5
    FastEstimator-Eval: step: 3750; epoch: 2; ce: 0.01713336; min_ce: 0.01713336; since_best: 0; accuracy: 0.9948; 
    FastEstimator-Finish: step: 3750; total_time: 16.72 sec; model_lr: 1.0001753e-06; 


## Model testing
`Estimator.test` triggers model testing with test dataset that specified in `Pipeline`. We can evaluate the model performance in the classification accuracy. 


```python
estimator.test()
```

    FastEstimator-Test: epoch: 2; accuracy: 0.9892; 


## Images inference 
In this step we run image inference directly using the model that just trained. 
We randomly select 5 images from testing dataset and infer them image by image with `Pipeline.transform` and `Netowork.transform`


```python
sample_num = 5

fig, axs = plt.subplots(sample_num, 3, figsize=(12,12))

axs[0,0].set_title("pipeline input")
axs[0,1].set_title("pipeline output")
axs[0,2].set_title("predict result")

for i, j in enumerate(np.random.randint(low=0, high=batch_size-1, size=sample_num)):
    data = {"x": test_data["x"][j]}
    axs[i,0].imshow(data["x"], cmap="gray")
    
    # run the pipeline
    data = pipeline.transform(data, mode="infer") 
    img = data["x_out"].squeeze(axis=(0,3))
    axs[i,1].imshow(img, cmap="gray")
    
    # run the network
    data = network.transform(data, mode="infer")
    predict = data["y_pred"].numpy().squeeze(axis=(0))
    axs[i,2].text(0.2, 0.5, "predicted number: {}".format(np.argmax(predict)))
    axs[i,2].get_xaxis().set_visible(False)
    axs[i,2].get_yaxis().set_visible(False)
```


![png](assets/example/image_classification/mnist_files/mnist_20_0.png)

