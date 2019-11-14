

### load_and_gradcam
```python
load_and_gradcam(model_path, input_paths, layer_id=None, dictionary_path=None, strip_alpha=False, save=False, save_dir=None)
```
A helper class to load input and invoke the gradcam api

#### Args:

* **model_path** :  The path the model file (str)
* **input_paths** :  The paths to model input files [(str),...] or to a folder of inputs [(str)]
* **layer_id** :  The layer id to be used. None defaults to the last conv layer in the model
* **dictionary_path** :  The path to a dictionary file encoding a 'class_idx'->'class_name' mapping
* **strip_alpha** :  Whether to collapse alpha channels when loading an input (bool)
* **save** :  Whether to save (True) or display (False) the resulting image
* **save_dir** :  Where to save the image if save=True