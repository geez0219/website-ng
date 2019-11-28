# Tutorial 11: Interpretation
___

In this tutorial we introduce the interpretation module, which provides various mechanisms for users to visualize / interpret what a neural network is seeing / focusing on within an input in order to make its decision. Since it will demonstrate bash commands which require certain path information, __this notebook should be run with the tutorial folder as the root directory__.

All interpretation modules may be invoked in three ways:   
    1) From a python API: visualize_COMMAND()    
    2) From the command line: fastestimator visualize COMMAND      
    3) From a __Trace__ which may then be combined with the __TensorBoard__ __Trace__ or any other image IO __Trace__

## Download a sample model for demonstration


```python
import tensorflow as tf
import os

# The following five lines are used to accommodate our test scripts
from IPython import get_ipython
if get_ipython() is not None:
    import matplotlib
    if matplotlib.get_backend() != 'agg':
      get_ipython().run_line_magic('matplotlib', 'inline')

model = tf.keras.applications.InceptionV3(weights='imagenet')  # This will download a .h5 file to ~/.keras/models
os.makedirs('./outputs', exist_ok=True)
model.save('./outputs/inceptionV3.h5')
```

## Interpretation with Python API

We'll start by running gradcam interpretation via the FastEstimator python API. We'll start by loading the pirate image into memory and sizing it appropriately. For the next few examples we'll be considering an image of a pirate ship: <div>&quot;<a href='https://www.flickr.com/photos/torley/3104607205/' target='_blank'>pirate ship by teepunch Jacobus</a>&quot;&nbsp;(<a rel='license' href='https://creativecommons.org/licenses/by-sa/2.0/' target='_blank'>CC BY-SA 2.0</a>)&nbsp;by&nbsp;<a xmlns:cc='http://creativecommons.org/ns#' rel='cc:attributionURL' property='cc:attributionName' href='https://www.flickr.com/people/torley/' target='_blank'>TORLEY</a></div>


```python
from fastestimator.util.util import load_dict, load_image

input_type = model.input.dtype
input_shape = model.input.shape
n_channels = 0 if len(input_shape) == 3 else input_shape[3]
input_height = input_shape[1]
input_width = input_shape[2]
inputs = [load_image('./image/pirates.jpg', channels=n_channels)]
tf_image = tf.stack([
    tf.image.resize_with_pad(tf.convert_to_tensor(im, dtype=input_type),
                             input_height,
                             input_width,
                             method='lanczos3') for im in inputs
])
pirates = tf.clip_by_value(tf_image, -1, 1)
dic = load_dict('./image/imagenet_class_index.json')
```

Now lets run the FE saliency api:


```python
from fastestimator.xai import visualize_gradcam

visualize_gradcam(inputs=pirates, model=model, decode_dictionary=dic, save_path=None, colormap=5)
```


![png](assets/tutorial/t11_interpretation_files/t11_interpretation_7_0.png)


Here we can see that there seems to be strong emphasis on the central mast of the ship, with some focus on the sails, but none on the pirate flag. It seems likely that the neural network has not learned that pirates are regular boats with a flag modifier, but rather that it has correlated a certain stereotypical ship design with pirate ships. On the other hand, the network does at least display good object localization - although there is some unnecessary focus on the sky in front of the ship. 

## Interpretation with Bash

Let's now consider the same image using caricatures. We'll start by running the caricature interpretation via the command line. Depending on how you downloaded FastEstimator, you may need to run `python setup.py install` from the parent directory in order to invoke the fastestimator command. To do this, delete the leading "#" character from the next cell and then run the cell. 


```python
#!fastestimator visualize caricature ./outputs/inceptionV3.h5 ./image/pirates.jpg --layers 196 --dictionary ./image/imagenet_class_index.json --save ./outputs
```

If you prefer to run the command using the python API, you can do the following instead:


```python
from fastestimator.xai import visualize_caricature

visualize_caricature(model=model, model_input=pirates, layer_ids=[196], decode_dictionary=dic, save_path="./outputs/")
```

    Layer 196: 100%|██████████| 512/512 [05:33<00:00,  1.73it/s]


    Saving to ./outputs/caricatures.png



![png](assets/tutorial/t11_interpretation_files/t11_interpretation_12_2.png)


Now lets load the image generated from the bash command back into memory for visualization:


```python
import matplotlib as mpl
import matplotlib.pyplot as plt
from fastestimator.xai import show_image

caricature = plt.imread('./outputs/caricatures.png')
mpl.rcParams['figure.dpi']=300
show_image(caricature)
plt.show()
```


![png](assets/tutorial/t11_interpretation_files/t11_interpretation_14_0.png)


A human who was trying to identify a pirate ship would likely focus on 2 key components: whether they are looking at a ship, and whether that ship is flying a pirate flag. From the caricature we can see that the stern of the ship is strongly emphasized, and that there is a large but ambiguously structured area for the sails. Interestingly, the pirate flag is completely absent from the caricature -- perhaps indicating the network is not interested in the flag.

## Interpretation with Traces

We now move on to a discussion of interpretation with Traces. For this example we will switch to the CIFAR10 dataset and see how UMAPs can be used to visualize what a network 'knows' about different classes.


```python
import tensorflow as tf

from fastestimator import Pipeline, Network, build, Estimator
from fastestimator.architecture import LeNet
from fastestimator.op.tensorop import Minmax, ModelOp, SparseCategoricalCrossentropy
from fastestimator.trace import UMap, ConfusionMatrix, VisLogger


(x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.cifar10.load_data()
data = {"train": {"x": x_train, "y": y_train}, "eval": {"x": x_eval, "y": y_eval}}
num_classes = 10
class_dictionary = {
    0: "airplane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

pipeline = Pipeline(batch_size=32, data=data, ops=Minmax(inputs="x", outputs="x"))

model = build(model_def=lambda: LeNet(input_shape=x_train.shape[1:], classes=num_classes),
              model_name="LeNet",
              optimizer="adam",
              loss_name="loss")

network = Network(ops=[
    ModelOp(inputs="x", model=model, outputs="y_pred"),
    SparseCategoricalCrossentropy(y_true="y", y_pred="y_pred", outputs="loss")
])

traces = [
    UMap(model_name="LeNet", model_input="x", labels="y", label_dictionary=class_dictionary),
    ConfusionMatrix("y", "y_pred", num_classes),
    VisLogger(show_images="LeNet_UMap")
]

estimator = Estimator(network=network, pipeline=pipeline, traces=traces, epochs=5, log_steps=750)

```


```python
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 0; total_train_steps: 7810; LeNet_lr: 0.001; 
    FastEstimator-Train: step: 0; loss: 2.294075; 
    FastEstimator-Train: step: 750; loss: 1.5830178; examples/sec: 1450.1; progress: 9.6%; 
    FastEstimator-Train: step: 1500; loss: 1.3869791; examples/sec: 1313.9; progress: 19.2%; 
    FastEstimator-Eval: step: 1562; epoch: 0; loss: 1.1841127; min_loss: 1.1841127; since_best_loss: 0; 
    confusion_matrix:
    [[664, 44, 70, 17, 18, 15, 17, 13,107, 33],
     [ 31,788,  9, 12,  3,  7, 12,  6, 57, 75],
     [ 94, 27,332,110, 64,192, 93, 43, 35,  9],
     [ 29, 21, 44,369, 44,333, 80, 38, 30, 12],
     [ 46, 13,138, 99,317,105,141,113, 21,  5],
     [ 19, 14, 44,144, 28,637, 29, 59, 15,  8],
     [ 13, 17, 36,103, 35, 65,690, 16, 15,  7],
     [ 28, 13, 17, 66, 44,152, 18,628,  8, 24],
     [170, 65, 13, 11,  2, 17,  7,  7,677, 30],
     [ 48,223,  5, 17,  4, 17, 24, 26, 55,579]];



![png](assets/tutorial/t11_interpretation_files/t11_interpretation_18_1.png)


    FastEstimator-Train: step: 2250; loss: 0.8873253; examples/sec: 1349.3; progress: 28.8%; 
    FastEstimator-Train: step: 3000; loss: 1.092779; examples/sec: 1342.3; progress: 38.4%; 
    FastEstimator-Eval: step: 3124; epoch: 1; loss: 1.0841651; min_loss: 1.0841651; since_best_loss: 0; 
    confusion_matrix:
    [[665, 48, 70,  8, 28,  4,  2, 33, 69, 71],
     [ 21,825,  4,  9,  8,  4,  6, 15,  9, 97],
     [ 67,  9,472, 85,139, 98, 19, 71, 17, 23],
     [ 29, 17, 58,442,104,188, 23, 81, 20, 36],
     [ 19, 11,100, 54,578, 37,  9,168, 15,  9],
     [ 13,  6, 58,209, 55,524,  4,103,  9, 18],
     [ 18, 18, 77,107,173, 65,437, 43,  4, 55],
     [ 12,  7, 18, 50, 55, 50,  1,773,  3, 30],
     [ 98, 66, 11, 20, 12,  5,  1, 13,702, 69],
     [ 17,162,  6,  5,  9,  7,  5, 26, 21,740]];



![png](assets/tutorial/t11_interpretation_files/t11_interpretation_18_3.png)


    FastEstimator-Train: step: 3750; loss: 0.8627761; examples/sec: 1307.7; progress: 48.0%; 
    FastEstimator-Train: step: 4500; loss: 1.2208309; examples/sec: 1269.2; progress: 57.6%; 
    FastEstimator-Eval: step: 4686; epoch: 2; loss: 0.9700292; min_loss: 0.9700292; since_best_loss: 0; 
    confusion_matrix:
    [[761, 70, 40, 11, 13, 10,  8,  9, 48, 28],
     [ 14,896,  3,  9,  3,  0,  9,  1,  5, 59],
     [ 98, 22,425, 75,106,101, 93, 42, 23, 14],
     [ 21, 26, 52,480, 82,159, 78, 49, 29, 21],
     [ 41, 19, 44, 59,617, 32, 92, 76, 12,  8],
     [ 21, 17, 40,183, 61,546, 34, 71, 14, 10],
     [ 12, 17, 30, 72, 32, 19,798,  8,  5,  7],
     [ 32, 15, 23, 38, 85, 45,  9,731,  3, 16],
     [144,120,  7, 12,  6,  4,  4,  8,652, 40],
     [ 40,211,  3, 16,  8,  3, 14, 18, 10,677]];



![png](assets/tutorial/t11_interpretation_files/t11_interpretation_18_5.png)


    FastEstimator-Train: step: 5250; loss: 0.7245537; examples/sec: 1310.4; progress: 67.2%; 
    FastEstimator-Train: step: 6000; loss: 1.0272136; examples/sec: 1249.0; progress: 76.8%; 
    FastEstimator-Eval: step: 6248; epoch: 3; loss: 0.9330987; min_loss: 0.93309873; since_best_loss: 0; 
    confusion_matrix:
    [[663, 75, 33,  7, 21,  3,  8,  8, 96, 83],
     [  5,887,  0,  2,  4,  0,  6,  3,  8, 84],
     [ 82, 22,490, 28,137, 66, 88, 34, 23, 28],
     [ 17, 41, 66,334,121,184,104, 47, 36, 48],
     [ 17, 21, 49, 20,684, 32, 59, 74, 24, 18],
     [ 13, 21, 50, 93, 71,609, 34, 69, 13, 25],
     [  6, 25, 22, 23, 67, 22,804,  4,  8, 18],
     [ 22, 13, 24, 16, 81, 45,  6,746,  6, 39],
     [ 46, 95,  5, 10,  7,  5,  5,  1,778, 48],
     [ 13,152,  1,  2,  3,  4,  6,  7, 15,796]];



![png](assets/tutorial/t11_interpretation_files/t11_interpretation_18_7.png)


    FastEstimator-Train: step: 6750; loss: 0.9370275; examples/sec: 1304.7; progress: 86.4%; 
    FastEstimator-Train: step: 7500; loss: 0.7190203; examples/sec: 1308.4; progress: 96.0%; 
    FastEstimator-Eval: step: 7810; epoch: 4; loss: 0.915211; min_loss: 0.915211; since_best_loss: 0; 
    confusion_matrix:
    [[699, 10, 66, 24, 33, 11,  7, 54, 39, 54],
     [ 28,709,  9, 14, 12,  7, 19, 15, 10,177],
     [ 53,  3,534, 67,113, 88, 39, 82,  7, 10],
     [  5,  3, 56,470, 95,219, 54, 74,  8, 14],
     [ 13,  3, 54, 42,662, 46, 19,153,  5,  2],
     [  8,  1, 39,147, 57,633, 13, 89,  3,  9],
     [  7,  1, 43, 68, 93, 44,716, 21,  2,  4],
     [  7,  1, 20, 24, 39, 59,  3,835,  1, 10],
     [102, 46, 17, 42, 19, 13,  6, 26,686, 41],
     [ 18, 29,  5, 17,  5,  9,  8, 37, 15,856]];



![png](assets/tutorial/t11_interpretation_files/t11_interpretation_18_9.png)


    FastEstimator-Finish: step: 7810; total_time: 219.51 sec; LeNet_lr: 0.001; 



![png](assets/tutorial/t11_interpretation_files/t11_interpretation_18_11.png)



![png](assets/tutorial/t11_interpretation_files/t11_interpretation_18_12.png)


As the UMaps reveal, the network is learning to separate man-made objects from natural ones. It also becomes clear that classes like 'cars' and 'trucks' are more similar to one another than they are to 'airplaines'. This all lines up with human intuition, which can increase confidence that the model has learned a useful embedding. The UMAP also helps to identify classes which will likely prove problematic for the network. In this case, the 'bird' class seems to be spread all over the map and therefore is likely to be confused with other classes. This is born out by the confusion matrix, which shows that the network has only 534 correct classifications for birds (class 2) vs 600+ for most of the other classes. 

If you want the Trace to save its output into TensorBoard, simply add a Tensorboard Trace to the traces list like follows:


```python
from fastestimator.trace import TensorBoard

traces = [
    UMap(model_name="LeNet", model_input="x", labels="y", label_dictionary=class_dictionary),
    TensorBoard(write_images="LeNet_UMap")
]
```
