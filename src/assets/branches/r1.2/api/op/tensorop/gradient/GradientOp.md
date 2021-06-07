## GradientOp<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/tensorop/gradient/gradient.py/#L29-L95>View source on Github</a>
```python
GradientOp(
	finals: Union[str, List[str]],
	outputs: Union[str, List[str]],
	inputs: Union[NoneType, str, List[str]]=None,
	model: Union[NoneType, tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module]=None,
	mode: Union[NoneType, str, Iterable[str]]=None
)
```
Return the gradients of finals w.r.t. inputs.


<h3>Args:</h3>


* **finals**: The tensor(s) to compute gradients from.

* **outputs**: The key(s) under which to save the gradients.

* **inputs**: The tensor(s) to compute gradients with respect to, mutually exclusive with `model`.

* **model**: The model instance to compute gradients with respect to, mutually exclusive with `inputs`.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

