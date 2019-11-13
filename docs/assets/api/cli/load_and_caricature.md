

### load_and_caricature
```python
load_and_caricature(model_path, input_paths, dictionary_path=None, save=False, save_dir=None, strip_alpha=False, layer_ids=None, print_layers=False, n_steps=512, learning_rate=0.05, blur=1, cossim_pow=0.5, sd=0.01, fft=True, decorrelate=True, sigmoid=True)
```


#### Args:

* **model_path (str)** :  The path to a keras model to be inspected by the Caricature visualization
* **layer_ids (int, list)** :  The layer(s) of the model to be inspected by the Caricature visualization
* **input_paths (list)** :  Strings corresponding to image files to be visualized
* **dictionary_path (string)** :  A path to a dictionary mapping model outputs to class names
* **save (bool)** :  Whether to save (True) or display (False) the result
* **save_dir (str)** :  Where to save the image if save is True
* **strip_alpha (bool)** :  Whether to strip the alpha channel from input images
* **print_layers (bool)** :  Whether to skip visualization and instead just print out the available layers in a model                             (useful for deciding which layers you might want to caricature)
* **n_steps (int)** :  How many steps of optimization to run when computing caricatures (quality vs time trade)
* **learning_rate (float)** :  The learning rate of the caricature optimizer. Should be higher than usual
* **blur (float)** :  How much blur to add to images during caricature generation
* **cossim_pow (float)** :  How much should similarity in form be valued versus creative license
* **sd (float)** :  The standard deviation of the noise used to seed the caricature
* **fft (bool)** :  Whether to use fft space (True) or image space (False) to create caricatures
* **decorrelate (bool)** :  Whether to use an ImageNet-derived color correlation matrix to de-correlate                        colors in the caricature. Parameter has no effect on grey scale images.
* **sigmoid (bool)** :  Whether to use sigmoid (True) or clipping (False) to bound the caricature pixel values