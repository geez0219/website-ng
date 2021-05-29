## ToSepia<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/univariate/to_sepia.py/#L24-L41>View source on Github</a>
```python
ToSepia(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None
)
```
Convert an RGB image to sepia.


<h3>Args:</h3>


* **inputs**: Key(s) of images to be converted to sepia.

* **outputs**: Key(s) into which to write the sepia images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train". Image types: uint8, float32

