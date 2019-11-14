## Caricature
```python
Caricature(model_name, layer_ids, model_input, n_inputs=1, resample_inputs=False, output_key=None, im_freq=1, mode='eval', decode_dictionary=None, n_steps=128, learning_rate=0.05, blur=1, cossim_pow=0.5, sd=0.01, fft=True, decorrelate=True, sigmoid=True)
```


#### Args:

* **model_name (str)** :  The model to be inspected by the Caricature visualization
* **layer_ids (int, list)** :  The layer(s) of the model to be inspected by the Caricature visualization
* **model_input (str, tf.Tensor)** :  The input to the model, either a string key or the actual input tensor
* **n_inputs (int)** :  How many samples should be drawn from the input_key tensor for visualization
* **resample_inputs (bool)** :  Whether to re-sample inputs every im_freq iterations or use the same throughout training                            Can only be True if model_input is a string
* **output_key (str)** :  The name of the output to be written into the batch dictionary
* **im_freq (int)** :  Frequency (in epochs) during which visualizations should be generated
* **mode (str)** :  The mode ('train', 'eval') on which to run the trace
* **decode_dictionary (dict)** :  A dictionary mapping model outputs to class names
* **n_steps (int)** :  How many steps of optimization to run when computing caricatures (quality vs time trade)
* **learning_rate (float)** :  The learning rate of the caricature optimizer. Should be higher than usual
* **blur (float)** :  How much blur to add to images during caricature generation
* **cossim_pow (float)** :  How much should similarity in form be valued versus creative license
* **sd (float)** :  The standard deviation of the noise used to seed the caricature
* **fft (bool)** :  Whether to use fft space (True) or image space (False) to create caricatures
* **decorrelate (bool)** :  Whether to use an ImageNet-derived color correlation matrix to de-correlate colors in                             the caricature. Parameter has no effect on grey scale images.
* **sigmoid (bool)** :  Whether to use sigmoid (True) or clipping (False) to bound the caricature pixel values