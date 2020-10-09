## MotionBlur<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/numpyop/univariate/motion_blur.py/#L24-L46>View source on Github</a>
```python
MotionBlur(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	blur_limit: Union[int, Tuple[int, int]]=7
)
```
Motion Blur the image with a randomly-sized kernel.


<h3>Args:</h3>


* **inputs**: Key(s) of images to be modified.

* **outputs**: Key(s) into which to write the modified images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **blur_limit**: maximum kernel size for blurring the input image. Should be in the range [3, inf). Image types: uint8, float32

