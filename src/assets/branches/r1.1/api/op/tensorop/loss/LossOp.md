## LossOp<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/tensorop/loss/loss.py/#L20-L56>View source on Github</a>
```python
LossOp(
	inputs: Union[str, List[str]]=None,
	outputs: List[str]=None,
	mode: Union[NoneType, str, Iterable[str]]=None,
	average_loss: bool=True
)
```
Abstract base LossOp class.

A base class for loss operations. It can be used directly to perform value pass-through (see the adversarial
training showcase for an example of when this is useful).


<h3>Args:</h3>


* **inputs**: A tuple or list like: [<y_pred>, <y_true>].

* **outputs**: String key under which to store the computed loss.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **average_loss**: Whether to average the element-wise loss after the Loss Op.

