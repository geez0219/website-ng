## SaliencyMask
```python
SaliencyMask(model)
```
Base class for saliency masks. Alone, this class doesn't do anything.This code was adapted from https://github.com/PAIR-code/saliency to be compatible with TensorFlow 2.0

### get_smoothed_mask
```python
get_smoothed_mask(self, model_input, stdev_spread=0.15, nsamples=25, magnitude=True, **kwargs)
```


#### Args:

* **model_input** :  Input tensor, shaped for the model ex. (1, 299, 299, 3)
* **stdev_spread** :  Amount of noise to add to the input, as fraction of the                        total spread (x_max - x_min). Defaults to 15%.
* **nsamples** :  Number of samples to average across to get the smooth gradient.
* **magnitude** :  If true, computes the sum of squares of gradients instead of                     just the sum. Defaults to true.

#### Returns:
            A saliency mask that is smoothed with the SmoothGrad method        