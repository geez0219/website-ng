## XAiTrace
```python
XAiTrace(model_name, model_input, n_inputs=1, resample_inputs=False, output_key=None, im_freq=1, mode='eval')
```


#### Args:

* **model_name (str)** :  The model to be inspected by the visualization
* **model_input (str, tf.Tensor)** :  The input to the model, either a string key or the actual input tensor
* **n_inputs (int)** :  How many inputs to be collected and passed to the model (if model_input is a string)
* **resample_inputs (bool)** :  Whether to re-sample inputs every im_freq iterations or use the same throughout training                            Can only be True if model_input is a string
* **output_key (str)** :  The name of the output to be written into the batch dictionary
* **im_freq (int)** :  Frequency (in epochs) during which visualizations should be generated
* **mode (str)** :  The mode ('train', 'eval') on which to run the trace