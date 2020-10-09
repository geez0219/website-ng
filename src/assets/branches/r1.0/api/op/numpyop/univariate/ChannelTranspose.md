## ChannelTranspose<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/op/numpyop/univariate/channel_transpose.py/#L22-L43>View source on Github</a>
```python
ChannelTranspose(
	inputs: Union[str, Iterable[str], Callable],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	axes: List[int]=(2, 0, 1)
)
```
Transpose the data (for example to make it channel-width-height instead of width-height-channel).


<h3>Args:</h3>


* **inputs**: Key(s) of images to be modified.

* **outputs**: Key(s) into which to write the modified images.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **axes**: The permutation order.

