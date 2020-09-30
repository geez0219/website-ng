## UNet
```python
UNet(
	input_size: Tuple[int, int, int]=(1, 128, 128)
)
-> None
```
A standard UNet implementation in PyTorch.

This class is intentionally not @traceable (models and layers are handled by a different process).


#### Args:

* **input_size** :  The size of the input tensor (channels, height, width).