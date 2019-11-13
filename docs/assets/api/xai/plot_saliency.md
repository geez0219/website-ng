

### plot_saliency
```python
plot_saliency(model, model_input, baseline_input=None, decode_dictionary=None, color_map='inferno', smooth=7)
```
Displays or saves a saliency mask interpretation of the given input

#### Args:

* **model** :  A model to evaluate. Should be a classifier which takes the 0th axis as the batch axis
* **model_input** :  Input tensor, shaped for the model ex. (1, 299, 299, 3)
* **baseline_input** :  An example of what a blank model input would be.                    Should be a tensor with the same shape as model_input
* **decode_dictionary** :  A dictionary of "class_idx" -> "class_name" associations
* **color_map** :  The color map to use to visualize the saliency maps.                    Consider "Greys_r", "plasma", or "magma" as alternatives
* **smooth** :  The number of samples to use when generating a smoothed image