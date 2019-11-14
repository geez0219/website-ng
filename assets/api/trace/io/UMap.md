## UMap
```python
UMap(model_name, model_input, n_inputs=500, resample_inputs=False, layer_id=-2, output_key=None, im_freq=1, mode='eval', labels=None, label_dictionary=None, legend_loc='best', **umap_parameters)
```


#### Args:

* **model_name (str)** :  The model to be inspected by the visualization
* **model_input (str, tf.Tensor)** :  The input to the model, either a string key or the actual input tensor
* **layer_id (int, str)** :  Which layer to inspect. Defaults to the second-to-last layer
* **n_inputs (int)** :  How many inputs to be collected and passed to the model (if model_input is a string)
* **resample_inputs (bool)** :  Whether to re-sample inputs every im_freq iterations or use the same throughout training                            Can only be True if model_input is a string
* **output_key (str)** :  The name of the output to be written into the batch dictionary
* **im_freq (int)** :  Frequency (in epochs) during which visualizations should be generated
* **mode (str)** :  The mode ('train', 'eval') on which to run the trace
* **labels** :  The (optional) key of the classes corresponding to the inputs (used for coloring points)
* **label_dictionary** :  An (optional) dictionary mapping labels from the label vector to other representations
* **(ex. {0** : 'dog', 1'cat'})
* **legend_loc** :  The location of the legend, or 'off' to disable figure legends
 **umap_parameters :  Extra parameters to be passed to the umap algorithm, ex. n_neighbors, n_epochs, etc.