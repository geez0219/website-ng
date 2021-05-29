## update_model<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/update_model.py/#L21-L91>View source on Github</a>
```python
update_model(
	model: Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module],
	gradients: List[Union[tensorflow.python.framework.ops.Tensor, torch.Tensor]],
	defer: bool=False,
	deferred: Union[Dict[str, List[Callable[[], NoneType]]], NoneType]=None
)
-> None
```
Update `model` weights based on a given `gradients`.

This method can be used with TensorFlow models:
```python
m = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")
x = tf.ones((3, 28, 28, 1))  # (batch, height, width, channels)
y = tf.constant((1, 0, 1))
with tf.GradientTape(persistent=True) as tape:
    pred = fe.backend.feed_forward(m, x)  # [[~0.5, ~0.5], [~0.5, ~0.5], [~0.5, ~0.5]]
    loss = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=y)  # ~2.3
    gradients = fe.backend.get_gradient(target=loss, sources=m.trainable_variables, tape=tape)
    fe.backend.update_model(m, gradients=gradients)
```

This method can be used with PyTorch models:
```python
m = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")
x = torch.ones((3, 1, 28, 28))  # (batch, channels, height, width)
y = torch.tensor((1, 0, 1))
pred = fe.backend.feed_forward(m, x)  # [[~0.5, ~0.5], [~0.5, ~0.5], [~0.5, ~0.5]]
loss = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=y)  # ~2.3
gradients = fe.backend.get_gradient(target=loss,
                                    sources=[x for x in m.parameters() if x.requires_grad])

fe.backend.update_model(m, gradients=gradients)
```


<h3>Args:</h3>


* **model**: A neural network instance to update.

* **gradients**: A list of tensors to update the models.

* **defer**: If True, then the model update function will be stored into the `deferred` dictionary rather than applied immediately.

* **deferred**: A dictionary in which model update functions are stored. 

<h3>Raises:</h3>


* **ValueError**: If `model` is an unacceptable data type.

* **AssertionError**: If `model` doesn't have `current_optimizer` attribute

* **AssertionError**: If Pytorch `model.current_optimizer` doesn't have `scaler` attribute

