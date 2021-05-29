## RandomFog<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/univariate/random_fog.py/#L24-L54>View source on Github</a>
```python
RandomFog(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	fog_coef_lower: float=0.3,
	fog_coef_upper: float=1.0,
	alpha_coef: float=0.08
)
```
Add fog to an image.


<h3>Args:</h3>


* **inputs**: Key(s) of images to be modified.

* **outputs**: Key(s) into which to write the modified images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **fog_coef_lower**: Lower limit for fog intensity coefficient. Should be in the range [0, 1].

* **fog_coef_upper**: Upper limit for fog intensity coefficient. Should be in the range [0, 1].

* **alpha_coef**: Transparency of the fog circles. Should be in the range [0, 1]. Image types: uint8, float32

