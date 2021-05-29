## GaussianBlur<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/univariate/gaussian_blur.py/#L24-L51>View source on Github</a>
```python
GaussianBlur(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	blur_limit: Union[int, Tuple[int, int]]=7,
	sigma_limit: Union[float, Tuple[float, float]]=0.0
)
```
Blur the image with a Gaussian filter of random kernel size.


<h3>Args:</h3>


* **inputs**: Key(s) of images to be modified.

* **outputs**: Key(s) into which to write the modified images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **blur_limit**: Maximum Gaussian kernel size for blurring the input image. Should be odd and in range [3, inf).

* **sigma_limit**: Gaussian kernel standard deviation. Must be greater in range [0, inf). If set single value sigma_limit will be in range (0, sigma_limit). If set to 0 sigma will be computed as sigma = 0.3*((ksize-1)*0.5 - 1) Image types: uint8, float32

