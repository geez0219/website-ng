## UNet<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/architecture/pytorch/unet.py/#L78-L135>View source on Github</a>
```python
UNet(
	input_size: Tuple[int, int, int]=(1, 128, 128)
)
-> None
```
A standard UNet implementation in PyTorch.

This class is intentionally not @traceable (models and layers are handled by a different process).


<h3>Args:</h3>


* **input_size**: The size of the input tensor (channels, height, width). 

<h3>Raises:</h3>


* **ValueError**: Length of `input_size` is not 3.

* **ValueError**: `input_size`[1] or `input_size`[2] is not a multiple of 16.

