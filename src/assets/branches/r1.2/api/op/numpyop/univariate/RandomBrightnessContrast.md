## RandomBrightnessContrast<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/univariate/random_brightness_contrast.py/#L24-L56>View source on Github</a>
```python
RandomBrightnessContrast(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	brightness_limit: Union[float, Tuple[float, float]]=0.2,
	contrast_limit: Union[float, Tuple[float, float]]=0.2,
	brightness_by_max: bool=True
)
```
Randomly change the brightness and contrast of an image.


<h3>Args:</h3>


* **inputs**: Key(s) of images to be modified.

* **outputs**: Key(s) into which to write the modified images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **brightness_limit**: Factor range for changing brightness. If limit is a single float, the range will be (-limit, limit).

* **contrast_limit**: Factor range for changing contrast. If limit is a single float, the range will be (-limit, limit).

* **brightness_by_max**: If True adjust contrast by image dtype maximum, else adjust contrast by image mean. Image types: uint8, float32

