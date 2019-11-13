## Saliency
```python
Saliency(model_name, model_input, n_inputs=1, resample_inputs=False, output_key=None, im_freq=1, mode='eval', label_dictionary=None, baseline_constant=0, color_map='inferno', smooth=7)
```


#### Args:

* **model_name (str)** :  The model to be inspected by the Saliency visualization
* **model_input (str, tf.Tensor)** :  The input to the model, either a string key or the actual input tensor
* **n_inputs (int)** :  How many samples should be drawn from the input_key tensor for visualization
* **resample_inputs (bool)** :  Whether to re-sample inputs every im_freq iterations or use the same throughout training                            Can only be True if model_input is a string
* **output_key (str)** :  The name of the output to be written into the batch dictionary
* **im_freq (int)** :  Frequency (in epochs) during which visualizations should be generated
* **mode (str)** :  The mode ('train', 'eval') on which to run the trace
* **label_dictionary (dict)** :  A dictionary of "class_idx" -> "class_name" associations
* **baseline_constant (float)** :  What constant value would a blank tensor have
* **color_map (str)** :  The color map to use to visualize the saliency maps.                     Consider "Greys_r", "plasma", or "magma" as alternatives
* **smooth (int)** :  The number of samples to use when generating a smoothed image