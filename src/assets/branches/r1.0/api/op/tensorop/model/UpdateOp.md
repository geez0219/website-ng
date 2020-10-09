## UpdateOp<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/op/tensorop/model/update.py/#L27-L54>View source on Github</a>
```python
UpdateOp(
	model: Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module],
	loss_name: str,
	mode: Union[NoneType, str, Iterable[str]]='train'
)
```
This class performs updates to a model's weights based on the loss.


<h3>Args:</h3>


* **model**: Model instance compiled by fe.build.

* **loss_name**: The name of loss.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

