## RandomGamma<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/numpyop/univariate/random_gamma.py/#L24-L48>View source on Github</a>
```python
RandomGamma(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	gamma_limit: Union[float, Tuple[float, float]]=(80, 120),
	eps: float=1e-07
)
```
Apply a gamma transform to an image.


<h3>Args:</h3>

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **gamma_limit** :  If gamma_limit is a single float value, the range will be (-gamma_limit, gamma_limit).
* **eps** :  A numerical stability constant to avoid division by zero.
* **Image types** :     uint8, float32



