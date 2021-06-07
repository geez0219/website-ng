## MeanSquaredError<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/tensorop/loss/mean_squared_error.py/#L29-L53>View source on Github</a>
```python
MeanSquaredError(
	inputs: Union[Tuple[str, str], List[str]],
	outputs: str,
	mode: Union[NoneType, str, Iterable[str]]='!infer',
	average_loss: bool=True
)
```
Calculate the mean squared error loss between two tensors.


<h3>Args:</h3>


* **inputs**: A tuple or list like: [<y_pred>, <y_true>].

* **outputs**: String key under which to store the computed loss.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **average_loss**: Whether to average the element-wise loss after the Loss Op.

