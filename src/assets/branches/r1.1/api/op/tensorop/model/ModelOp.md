## ModelOp<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/tensorop/model/model.py/#L29-L63>View source on Github</a>
```python
ModelOp(
	model: Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module],
	inputs: Union[NoneType, str, Iterable[str]]=None,
	outputs: Union[NoneType, str, Iterable[str]]=None,
	mode: Union[NoneType, str, Iterable[str]]=None,
	trainable: bool=True
)
```
This class performs forward passes of a neural network over batch data to generate predictions.


<h3>Args:</h3>

* **model** :  A model compiled by fe.build.
* **inputs** :  String key of input training data.
* **outputs** :  String key under which to store predictions.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **trainable** :  Indicates whether the model should have its weights tracked for update.



