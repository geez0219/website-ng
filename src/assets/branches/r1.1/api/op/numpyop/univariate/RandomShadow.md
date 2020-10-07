## RandomShadow<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/numpyop/univariate/random_shadow.py/#L24-L63>View source on Github</a>
```python
RandomShadow(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	shadow_roi: Tuple[float, float, float, float]=(0.0, 0.5, 1.0, 1.0),
	num_shadows_lower: int=1,
	num_shadows_upper: int=2,
	shadow_dimension: int=5
)
```
Add shadows to an image


<h3>Args:</h3>

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **shadow_roi** :  Region of the image where shadows will appear (x_min, y_min, x_max, y_max).        All values should be in range [0, 1].
* **num_shadows_lower** :  Lower limit for the possible number of shadows. Should be in range [0, `num_shadows_upper`].
* **num_shadows_upper** :  Lower limit for the possible number of shadows.        Should be in range [`num_shadows_lower`, inf].
* **shadow_dimension** :  Number of edges in the shadow polygons.
* **Image types** :     uint8, float32



