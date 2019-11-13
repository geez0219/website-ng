## GradCam
```python
GradCam(model_name, model_input, n_inputs=1, resample_inputs=False, layer_id=None, output_key=None, im_freq=1, mode='eval', label_dictionary=None, color_map=14)
```
Draw GradCam heatmaps for given inputs

#### Args:

* **model_name (str)** :  The model to be inspected by the visualization
* **model_input (str, tf.Tensor)** :  The input to the model, either a string key or the actual input tensor
* **n_inputs (int)** :  How many inputs to be collected and passed to the model (if model_input is a string)
* **resample_inputs (bool)** :  Whether to re-sample inputs every im_freq iterations or use the same throughout training                            Can only be True if model_input is a string
* **output_key (str)** :  The name of the output to be written into the batch dictionary
* **im_freq (int)** :  Frequency (in epochs) during which visualizations should be generated
* **mode (str)** :  The mode ('train', 'eval') on which to run the trace
* **layer_id (int, str, None)** :  Which layer to inspect. Should be a convolutional layer. If None, the last                                     acceptable layer from the model will be selected
* **label_dictionary (dict)** :  A dictionary of "class_idx" -> "class_name" associations
* **color_map (int)** :  Which colormap to use when generating the heatmaps