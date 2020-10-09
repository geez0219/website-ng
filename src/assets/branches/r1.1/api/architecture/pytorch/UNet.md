## UNet<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/architecture/pytorch/unet.py/#L78-L120>View source on Github</a>
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

