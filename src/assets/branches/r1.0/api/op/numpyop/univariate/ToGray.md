## ToGray<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/op/numpyop/univariate/to_gray.py/#L22-L39>View source on Github</a>
```python
ToGray(
	inputs: Union[str, Iterable[str], Callable],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None
)
```
Convert an RGB image to grayscale. If the mean pixel value of the result is > 127, the image is inverted.


<h3>Args:</h3>

* **inputs** :  Key(s) of images to be converted to grayscale.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **Image types** :     uint8, float32



