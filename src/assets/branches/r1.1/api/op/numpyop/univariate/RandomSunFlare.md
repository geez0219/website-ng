## RandomSunFlare<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/numpyop/univariate/random_sun_flare.py/#L24-L69>View source on Github</a>
```python
RandomSunFlare(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	flare_roi: Tuple[float, float, float, float]=(0, 0, 1, 0.5),
	angle_lower: float=0.0,
	angle_upper: float=1.0,
	num_flare_circles_lower: int=6,
	num_flare_circles_upper: int=10,
	src_radius: int=400,
	src_color: Tuple[int, int, int]=(255, 255, 255)
)
```
Add a sun flare to the image.


<h3>Args:</h3>


* **inputs**: Key(s) of images to be normalized.

* **outputs**: Key(s) into which to write the modified images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **flare_roi**: region of the image where flare will appear (x_min, y_min, x_max, y_max). All values should be in range [0, 1].

* **angle_lower**: should be in range [0, `angle_upper`].

* **angle_upper**: should be in range [`angle_lower`, 1].

* **num_flare_circles_lower**: lower limit for the number of flare circles. Should be in range [0, `num_flare_circles_upper`].

* **num_flare_circles_upper**: upper limit for the number of flare circles. Should be in range [`num_flare_circles_lower`, inf].

* **src_radius**: Radius of the flare.

* **src_color**: Color of the flare (R,G,B). Image types: uint8, float32

