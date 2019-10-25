

### RetinaNet
```python
RetinaNet(input_shape, num_classes, num_anchor=9)
```
Creates the RetinaNet. RetinaNet is composed of an FPN, a classification sub-network and a localizationregression sub-network.

#### Args:

* **input_shape (tuple)** :  shape of input image.
* **num_classes (int)** :  number of classes.
* **num_anchor (int, optional)** :  number of anchor boxes. Defaults to 9.

#### Returns:

* **'Model' object** :  RetinaNet.