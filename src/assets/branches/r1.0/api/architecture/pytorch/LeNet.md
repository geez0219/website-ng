## LeNet
```python
LeNet(input_shape:Tuple[int, int, int]=(1, 28, 28), classes:int=10) -> None
```
A standard LeNet implementation in pytorch.    The LeNet model has 3 convolution layers and 2 dense layers.

#### Args:

* **input_shape** :  The shape of the model input (channels, height, width).
* **classes** :  The number of outputs the model should generate.    