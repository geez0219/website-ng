## LeNet<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/architecture/tensorflow/lenet.py/#L22-L43>View source on Github</a>
```python
LeNet(
	input_shape: Tuple[int, int, int]=(28, 28, 1),
	classes: int=10
)
-> tensorflow.python.keras.engine.training.Model
```
A standard LeNet implementation in TensorFlow.

The LeNet model has 3 convolution layers and 2 dense layers.


<h3>Args:</h3>


* **input_shape**: shape of the input data (height, width, channels).

* **classes**: The number of outputs the model should generate. 

<h3>Returns:</h3>

<ul class="return-block"><li>    A TensorFlow LeNet model.</li></ul>

