## MultiplicativeNoise<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/op/numpyop/univariate/multiplicative_noise.py/#L22-L54>View source on Github</a>
```python
MultiplicativeNoise(
	inputs: Union[str, Iterable[str], Callable],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	multiplier: Union[float, Tuple[float, float]]=(0.9, 1.1),
	per_channel: bool=False,
	elementwise: bool=False
)
```
Multiply an image with random perturbations.


<h3>Args:</h3>


* **inputs**: Key(s) of images to be modified.

* **outputs**: Key(s) into which to write the modified images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **multiplier**: If a single float, the image will be multiplied by this number. If tuple of floats then `multiplier` will be in the range [multiplier[0], multiplier[1]).

* **per_channel**: Whether to sample different multipliers for each channel of the image.

* **elementwise**: If `False` multiply multiply all pixels in an image with a random value sampled once. If `True` Multiply image pixels with values that are pixelwise randomly sampled. Image types: uint8, float32

