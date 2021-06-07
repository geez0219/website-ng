## ResNet9<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/architecture/tensorflow/resnet9.py/#L22-L68>View source on Github</a>
```python
ResNet9(
	input_size: Tuple[int, int, int]=(32, 32, 3),
	classes: int=10
)
-> tensorflow.python.keras.engine.training.Model
```
A small 9-layer ResNet Tensorflow model for cifar10 image classification.
The model architecture is from https://github.com/davidcpage/cifar10-fast


<h3>Args:</h3>


* **input_size**: The size of the input tensor (height, width, channels).

* **classes**: The number of outputs the model should generate. 

<h3>Raises:</h3>


* **ValueError**: Length of `input_size` is not 3.

* **ValueError**: `input_size`[0] or `input_size`[1] is not a multiple of 16.



<h3>Returns:</h3>

<ul class="return-block"><li>    A TensorFlow ResNet9 model.</li></ul>

