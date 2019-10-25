

### load_and_saliency
```python
load_and_saliency(model_path, input_paths, baseline=-1, dictionary_path=None, strip_alpha=False, smooth_factor=7, save=False, save_dir=None)
```
A helper class to load input and invoke the saliency api

#### Args:

* **model_path** :  The path the model file (str)
* **input_paths** :  The paths to model input files [(str),...] or to a folder of inputs [(str)]
* **baseline** :  Either a number corresponding to the baseline for integration, or a path to a baseline file
* **dictionary_path** :  The path to a dictionary file encoding a 'class_idx'->'class_name' mapping
* **strip_alpha** :  Whether to collapse alpha channels when loading an input (bool)
* **smooth_factor** :  How many iterations of the smoothing algorithm to run (int)
* **save** :  Whether to save (True) or display (False) the resulting image
* **save_dir** :  Where to save the image if save=True