## RGBShift<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/numpyop/univariate/rgb_shift.py/#L24-L57>View source on Github</a>
```python
RGBShift(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	r_shift_limit: Union[int, Tuple[int, int]]=20,
	g_shift_limit: Union[int, Tuple[int, int]]=20,
	b_shift_limit: Union[int, Tuple[int, int]]=20
)
```
Randomly shift the channel values for an input RGB image.


<h3>Args:</h3>


* **inputs**: Key(s) of images to be normalized.

* **outputs**: Key(s) into which to write the normalized images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **r_shift_limit**: range for changing values for the red channel. If r_shift_limit is a single int, the range will be (-r_shift_limit, r_shift_limit).

* **g_shift_limit**: range for changing values for the green channel. If g_shift_limit is a single int, the range will be (-g_shift_limit, g_shift_limit).

* **b_shift_limit**: range for changing values for the blue channel. If b_shift_limit is a single int, the range will be (-b_shift_limit, b_shift_limit). Image types: uint8, float32

