

### generate_caricatures
```python
generate_caricatures(model, model_input, layer_id, n_steps=100, learning_rate=0.05, blur=1, cossim_pow=0.5, sd=0.01, fft=True, decorrelate=True, sigmoid=True)
```


#### Args:

* **model (model)** :  The keras model to be inspected by the Caricature visualization
* **model_input (tensor)** :  The input images to be fed to the model
* **layer_id (int)** :  The layer of the model to be inspected by the Caricature visualization
* **n_steps (int)** :  How many steps of optimization to run when computing caricatures (quality vs time trade)
* **learning_rate (float)** :  The learning rate of the caricature optimizer. Should be higher than usual
* **blur (float)** :  How much blur to add to images during caricature generation
* **cossim_pow (float)** :  How much should similarity in form be valued versus creative license
* **sd (float)** :  The standard deviation of the noise used to seed the caricature
* **fft (bool)** :  Whether to use fft space (True) or image space (False) to create caricatures
* **decorrelate (bool)** :  Whether to use an ImageNet-derived color correlation matrix to de-correlate                        colors in the caricature. Parameter has no effect on grey scale images.
* **sigmoid (bool)** :  Whether to use sigmoid (True) or clipping (False) to bound the caricature pixel values