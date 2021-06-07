## UpdateOp<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/tensorop/model/update.py/#L33-L254>View source on Github</a>
```python
UpdateOp(
	model: Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module],
	loss_name: str,
	gradients: Union[str, NoneType]=None,
	mode: Union[NoneType, str, Iterable[str]]='train',
	merge_grad: int=1,
	defer: bool=False
)
```
This class performs updates to a model's weights based on the loss.


<h3>Args:</h3>


* **model**: Model instance compiled by fe.build.

* **loss_name**: The input loss key.

* **gradients**: An optional key containing model gradients. These will be directly applied to the model weights during an update. If not provided, gradients will be computed based on the specified loss_name, which will automatically handle any desired mixed-precision scaling. This argument shouldn't be used if mixed-precision training is enabled.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **merge_grad**: The gradient accumulation times before model update. Ex: if `merge_grad` = 3, for every three Op calls only the third one updates the model. The first two calls only accumulate its gradients. This default value is 1 and it will update the model at every step.

* **defer**: Whether to defer the actual application of the update until the end of the step. This can be necessary in PyTorch when trying to update multiple models which depend on one another (ex. certain GANs). By default, all UpdateOps which appear contiguously as the last ops of a Network will be deferred. We hope that you will never need to worry about this flag, but it's here for you if you need it. Raise:

* **ValueError**: When model is mixed-precision and `gradients` is provided.

* **ValueError**: Network framework is not one of "tf" or "torch".

* **ValueError**: `merge_grad` is larger than 1 in multi-GPU configuration.

* **RuntimeError**: If attempting to modify a PyTorch model which relied on gradients within a different PyTorch model which has in turn already undergone a non-deferred update.

