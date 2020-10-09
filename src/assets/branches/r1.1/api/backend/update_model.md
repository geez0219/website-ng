## update_model<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/backend/update_model.py/#L25-L116>View source on Github</a>
```python
update_model(
	model: Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module],
	loss: Union[tensorflow.python.framework.ops.Tensor, torch.Tensor],
	tape: Union[tensorflow.python.eager.backprop.GradientTape, NoneType]=None,
	retain_graph: bool=True,
	scaler: Union[torch.cuda.amp.grad_scaler.GradScaler, NoneType]=None,
	defer: bool=False,
	deferred: Union[Dict[str, List[Callable[[], NoneType]]], NoneType]=None
)
-> None
```
Update `model` weights based on a given `loss`.

This method can be used with TensorFlow models:
```python
m = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")
x = tf.ones((3,28,28,1))  # (batch, height, width, channels)
y = tf.constant((1, 0, 1))
with tf.GradientTape(persistent=True) as tape:
    pred = fe.backend.feed_forward(m, x)  # [[~0.5, ~0.5], [~0.5, ~0.5], [~0.5, ~0.5]]
    loss = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=y)  # ~2.3
    fe.backend.update_model(m, loss=loss, tape=tape)
```

This method can be used with PyTorch models:
```python
m = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")
x = torch.ones((3,1,28,28))  # (batch, channels, height, width)
y = torch.tensor((1, 0, 1))
pred = fe.backend.feed_forward(m, x)  # [[~0.5, ~0.5], [~0.5, ~0.5], [~0.5, ~0.5]]
loss = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=y)  # ~2.3
fe.backend.update_model(m, loss=loss)
```


<h3>Args:</h3>


* **model**: A neural network instance to update.

* **loss**: A loss value to compute gradients from.

* **tape**: A TensorFlow GradientTape which was recording when the `loss` was computed (iff using TensorFlow).

* **retain_graph**: Whether to keep the model graph in memory (applicable only for PyTorch).

* **scaler**: A PyTorch loss scaler that scales loss when PyTorch mixed precision is used.

* **defer**: If True, then the model update function will be stored into the `deferred` dictionary rather than applied immediately.

* **deferred**: A dictionary in which model update functions are stored. 

<h3>Raises:</h3>


* **ValueError**: If `model` is an unacceptable data type.

* **RuntimeError**: If attempting to modify a PyTorch model which relied on gradients within a different PyTorch model
        which has in turn already undergone a non-deferred update.

