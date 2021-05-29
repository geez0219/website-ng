## ResNet9<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/architecture/pytorch/resnet9.py/#L22-L89>View source on Github</a>
```python
ResNet9(
	input_size: Tuple[int, int, int]=[3, 32, 32],
	classes: int=10
)
```
A 9-layer ResNet PyTorch model for cifar10 image classification.
The model architecture is from https://github.com/davidcpage/cifar10-fast


<h3>Args:</h3>


* **input_size**: The size of the input tensor (channels, height, width). Both width and height of input_size should not be smaller than 16.

* **classes**: The number of outputs. 

<h3>Raises:</h3>


* **ValueError**: Length of `input_size` is not 3.

* **ValueError**: `input_size`[1] or `input_size`[2] is not a multiple of 16.

