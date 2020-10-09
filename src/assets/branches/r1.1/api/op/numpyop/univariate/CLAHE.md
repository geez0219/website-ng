## CLAHE<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/numpyop/univariate/clahe.py/#L24-L49>View source on Github</a>
```python
CLAHE(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	clip_limit: Union[float, Tuple[float, float]]=4.0,
	tile_grid_size: Tuple[int, int]=(8, 8)
)
```
Apply contrast limited adaptive histogram equalization to the image.


<h3>Args:</h3>


* **inputs**: Key(s) of images to be modified.

* **outputs**: Key(s) into which to write the modified images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **clip_limit**: upper threshold value for contrast limiting. If clip_limit is a single float value, the range will be (1, clip_limit).

* **tile_grid_size**: size of grid for histogram equalization. Image types: uint8

