# Advanced Tutorial 8: Explainable AI (XAI)

## Overview
In this tutorial, we will discuss the following topics:
* [Saliency Maps](./tutorials/r1.1/advanced/t08_xai#ta08saliency)
    * [With Traces](./tutorials/r1.1/advanced/t08_xai#ta08with)
    * [Without Traces](./tutorials/r1.1/advanced/t08_xai#ta08without)

<a id='ta08saliency'></a>

## Saliency Maps

Suppose you have a neural network that is performing image classification. The network tells you that the image it is looking at is an airplane, but you want to know whether it is really detecting an airplane, or if it is 'cheating' by noticing the blue sky in the image background. To answer this question, all you need to do is add the `Saliency` `Trace` to your list of traces, and pass its output to one of either the `ImageSaver`, `ImageViewer`, or `TensorBoard` `Traces`.

<a id='ta08with'></a>


```python
import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.backend import squeeze
from fastestimator.dataset.data import cifar10
from fastestimator.op.numpyop.univariate import Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver, ImageViewer, TensorBoard
from fastestimator.trace.metric import Accuracy
from fastestimator.trace.xai import Saliency
from fastestimator.util import to_number

import matplotlib.pyplot as plt
import numpy as np

label_mapping = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}

batch_size=32

train_data, eval_data = cifar10.load_data()
test_data = eval_data.split(0.5)
pipeline = fe.Pipeline(
    train_data=train_data,
    eval_data=eval_data,
    test_data=test_data,
    batch_size=batch_size,
    ops=[Normalize(inputs="x", outputs="x")],
    num_process=0)

model = fe.build(model_fn=lambda: LeNet(input_shape=(32, 32, 3)), optimizer_fn="adam")
network = fe.Network(ops=[
    ModelOp(model=model, inputs="x", outputs="y_pred"),
    CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
    UpdateOp(model=model, loss_name="ce")
])

traces = [
    Accuracy(true_key="y", pred_key="y_pred"),
    LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=3750, init_lr=1e-3)),
    Saliency(model=model,
             model_inputs="x",
             class_key="y",
             model_outputs="y_pred",
             samples=5,
             label_mapping=label_mapping),
    ImageViewer(inputs="saliency")
]
estimator = fe.Estimator(pipeline=pipeline,
                         network=network,
                         epochs=5,
                         traces=traces,
                         log_steps=1000)
```

In this example we will be using the `ImageViewer` `Trace`, since it will allow us to visualize the outputs within this Notebook. If you wanted your images to appear in TensorBoard, simply construct a `TensorBoard` `Trace` with the "write_images" argument set to "saliency". 


```python
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; num_device: 1; logging_interval: 1000; 
    FastEstimator-Train: step: 1; ce: 2.4207764; model_lr: 0.0009999998; 
    FastEstimator-Train: step: 1000; ce: 1.13122; steps/sec: 359.79; model_lr: 0.00083473074; 
    FastEstimator-Train: step: 1563; epoch: 1; epoch_time: 4.91 sec; 



    
![png](assets/branches/r1.1/tutorial/advanced/t08_xai_files/t08_xai_6_1.png)
    


    FastEstimator-Eval: step: 1563; epoch: 1; ce: 1.1183364; accuracy: 0.6022; 
    FastEstimator-Train: step: 2000; ce: 1.0117686; steps/sec: 349.3; model_lr: 0.00044828805; 
    FastEstimator-Train: step: 3000; ce: 0.84499186; steps/sec: 362.07; model_lr: 9.639601e-05; 
    FastEstimator-Train: step: 3126; epoch: 2; epoch_time: 4.29 sec; 



    
![png](assets/branches/r1.1/tutorial/advanced/t08_xai_files/t08_xai_6_3.png)
    


    FastEstimator-Eval: step: 3126; epoch: 2; ce: 0.9405009; accuracy: 0.674; 
    FastEstimator-Train: step: 4000; ce: 0.9253516; steps/sec: 385.52; model_lr: 0.0009890847; 
    FastEstimator-Train: step: 4689; epoch: 3; epoch_time: 4.11 sec; 



    
![png](assets/branches/r1.1/tutorial/advanced/t08_xai_files/t08_xai_6_5.png)
    


    FastEstimator-Eval: step: 4689; epoch: 3; ce: 0.94179326; accuracy: 0.6724; 
    FastEstimator-Train: step: 5000; ce: 0.66330093; steps/sec: 375.63; model_lr: 0.00075025; 
    FastEstimator-Train: step: 6000; ce: 1.046946; steps/sec: 370.67; model_lr: 0.000346146; 
    FastEstimator-Train: step: 6252; epoch: 4; epoch_time: 4.2 sec; 



    
![png](assets/branches/r1.1/tutorial/advanced/t08_xai_files/t08_xai_6_7.png)
    


    FastEstimator-Eval: step: 6252; epoch: 4; ce: 0.8270567; accuracy: 0.7174; 
    FastEstimator-Train: step: 7000; ce: 0.6407217; steps/sec: 357.48; model_lr: 4.4184046e-05; 
    FastEstimator-Train: step: 7815; epoch: 5; epoch_time: 4.4 sec; 



    
![png](assets/branches/r1.1/tutorial/advanced/t08_xai_files/t08_xai_6_9.png)
    


    FastEstimator-Eval: step: 7815; epoch: 5; ce: 0.9042755; accuracy: 0.6878; 
    FastEstimator-Finish: step: 7815; total_time: 30.5 sec; model_lr: 0.0009827082; 



```python
estimator.test()
```


    
![png](assets/branches/r1.1/tutorial/advanced/t08_xai_files/t08_xai_7_0.png)
    


    FastEstimator-Test: step: 7815; epoch: 5; accuracy: 0.69; 


In the images above, the 'saliency' column corresponds to a raw saliency mask generated by back-propagating a model's output prediction onto the input image. 'Smoothed saliency' combines multiple saliency masks for each image 'x', where each mask is generated by slightly perturbing the input 'x' before running the forward and backward gradient passes. The number of samples to be combined is controlled by the "smoothing" argument in the `Saliency` `Trace` constructor. 'Integrated saliency' is a saliency mask generated by starting from a baseline blank image and linearly interpolating the image towards 'x' over a number of steps defined by the "integrating" argument in the Saliency constructor. The resulting masks are then combined together. The 'SmInt Saliency' (Smoothed-Integrated) column combines smoothing and integration together. SmInt is generally considered to give the most reliable indication of the important features in an image, but it also takes the longest to compute. It is possible to disable the more complex columns by setting the 'smoothing' and 'integrating' parameters to 0. The 'x saliency' column shows the input image overlaid with whatever saliency column is furthest to the right (SmInt, unless that has been disabled).

<a id='ta08without'></a>

## Saliency Maps without Traces

Suppose that you want to generate Saliency masks without using a `Trace`. This can be done through the fe.xai package:


```python
import tempfile
import os

pipeline.batch_size = 6
batch = pipeline.get_results()
batch = fe.backend.to_tensor(batch, "tf")  # Convert the batch to TensorFlow

saliency_generator = fe.xai.SaliencyNet(model=model, model_inputs="x", model_outputs="y_pred")
images = saliency_generator.get_masks(batch=batch)

# Let's convert 'y' and 'y_pred' from numeric values to strings for readability:
val_to_label = {val: key for key, val in label_mapping.items()}
y = np.array([val_to_label[clazz] for clazz in to_number(squeeze(batch["y"]))])
y_pred = np.array([val_to_label[clazz] for clazz in to_number(squeeze(images["y_pred"]))])

# Now simply load up an ImgData object and let it handle laying out the final result for you
save_dir = tempfile.mkdtemp()
images = fe.util.ImgData(colormap="inferno", y=y, y_pred=y_pred, x=batch["x"], saliency=images["saliency"])
fig = images.paint_figure(save_path=os.path.join(save_dir, "t08a_saliency.png")) # save_path is optional, but a useful feature to know about
plt.show()
```


    
![png](assets/branches/r1.1/tutorial/advanced/t08_xai_files/t08_xai_11_0.png)
    


The `SaliencyNet` class also provides 'get_smoothed_masks' and 'get_integrated_masks' methods for generating the more complicated saliency maps. 
