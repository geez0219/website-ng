

### lossNet
```python
lossNet(input_shape=(256, 256, 3), styleLayers=['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3'], contentLayers=['block3_conv3'])
```
Creates the network to compute the style loss.This network outputs a dictionary with outputs values for style and content, based on a list of layers from VGG16for each.

#### Args:

* **input_shape (tuple, optional)** :  shape of input image. Defaults to (256, 256, 3).
* **styleLayers (list, optional)** :  list of style layers from VGG16. Defaults to ["block1_conv2", "block2_conv2",    "block3_conv3", "block4_conv3"].
* **contentLayers (list, optional)** :  list of content layers from VGG16. Defaults to ["block3_conv3"].

#### Returns:

* **'Model' object** :  style loss Network.