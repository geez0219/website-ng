## UpdateOp<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/tensorop/model/update.py/#L29-L79>View source on Github</a>
```python
UpdateOp(
	model: Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module],
	loss_name: str,
	mode: Union[NoneType, str, Iterable[str]]='train',
	defer: bool=False
)
```
This class performs updates to a model's weights based on the loss.


<h3>Args:</h3>

* **model** :  Model instance compiled by fe.build.
* **loss_name** :  The name of loss.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **defer** :  Whether to defer the actual application of the update until the end of the step. This can be necessary        in PyTorch when trying to update multiple models which depend on one another (ex. certain GANs). By default,        all UpdateOps which appear contiguously as the last ops of a Network will be deferred. We hope that you will        never need to worry about this flag, but it's here for you if you need it.



