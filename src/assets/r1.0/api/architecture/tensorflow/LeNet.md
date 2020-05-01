

### LeNet
```python
LeNet(input_shape:Tuple[int, int, int]=(28, 28, 1), classes:int=10) -> tensorflow.python.keras.engine.training.Model
```
A standard LeNet implementation in TensorFlow.The LeNet model has 3 convolution layers and 2 dense layers.

#### Args:

* **input_shape** :  shape of the input data (height, width, channels).
* **classes** :  The number of outputs the model should generate.

#### Returns:
    A TensorFlow LeNet model.