## ISONoise<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/univariate/iso_noise.py/#L24-L49>View source on Github</a>
```python
ISONoise(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	color_shift: Tuple[float, float]=(0.01, 0.05),
	intensity: Tuple[float, float]=(0.1, 0.5)
)
```
Apply camera sensor noise.


<h3>Args:</h3>


* **inputs**: Key(s) of images to be modified.

* **outputs**: Key(s) into which to write the modified images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **color_shift**: Variance range for color hue change. Measured as a fraction of 360 degree Hue angle in the HLS colorspace.

* **intensity**: Multiplicative factor that controls the strength of color and luminace noise. Image types: uint8

