## LeNet<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/architecture/pytorch/lenet.py/#L22-L56>View source on Github</a>
```python
LeNet(
	input_shape: Tuple[int, int, int]=(1, 28, 28),
	classes: int=10
)
-> None
```
A standard LeNet implementation in pytorch.

This class is intentionally not @traceable (models and layers are handled by a different process).

The LeNet model has 3 convolution layers and 2 dense layers.


<h3>Args:</h3>


* **input_shape**: The shape of the model input (channels, height, width).

* **classes**: The number of outputs the model should generate.

