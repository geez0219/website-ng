

### plot_gradcam
```python
plot_gradcam(inputs, model, layer_id=None, target_class=None, decode_dictionary=None, colormap=14)
```
Creates a GradCam interpretation of the given input

#### Args:

* **inputs (tf.tensor)** :  Model input, with batch along the zero axis
* **model (tf.keras.model)** :  tf.keras model to inspect
* **layer_id (int, str, None)** :  Which layer to inspect. Should be a convolutional layer. If None, the last                                     acceptable layer from the model will be selected
* **target_class (int, None)** :  Which output class to try to explain. None will default to explaining the maximum                                     likelihood prediction
* **decode_dictionary (dict)** :  A dictionary of "class_idx" -> "class_name" associations
* **colormap (int)** :  Which colormap to use when generating the heatmaps

#### Returns:
    The matplotlib figure handle