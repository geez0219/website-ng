## RandomRain<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/op/numpyop/univariate/random_rain.py/#L22-L67>View source on Github</a>
```python
RandomRain(
	inputs: Union[str, Iterable[str], Callable],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	slant_lower: int=-10,
	slant_upper: int=10,
	drop_length: int=20,
	drop_width: int=1,
	drop_color: Tuple[int, int, int]=(200, 200, 200),
	blur_value: int=7,
	brightness_coefficient: float=0.7,
	rain_type: Union[str, NoneType]=None
)
```
Add rain to an image


<h3>Args:</h3>


* **inputs**: Key(s) of images to be modified.

* **outputs**: Key(s) into which to write the modified images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **slant_lower**: Should be in range [-20, 20].

* **slant_upper**: Should be in range [-20, 20].

* **drop_length**: Should be in range [0, 100].

* **drop_width**: Should be in range [1, 5].

* **drop_color**: Rain lines color (r, g, b).

* **blur_value**: How blurry to make the rain.

* **brightness_coefficient**: Rainy days are usually shady. Should be in range [0, 1].

* **rain_type**: One of [None, "drizzle", "heavy", "torrential"]. Image types: uint8, float32

