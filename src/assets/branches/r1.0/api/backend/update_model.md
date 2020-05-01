

### update_model
```python
update_model(model:Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module], loss:Union[tensorflow.python.framework.ops.Tensor, torch.Tensor], tape:Union[tensorflow.python.eager.backprop.GradientTape, NoneType]=None, retain_graph:bool=True)
```
Update `model` weights based on a given `loss`.
* **This method can be used with TensorFlow models** : ```pythonm = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn="adam")x = tf.ones((3,28,28,1))  # (batch, height, width, channels)y = tf.constant((1, 0, 1))
* **with tf.GradientTape(persistent=True) as tape** :     pred = fe.backend.feed_forward(m, x)  # [[~0.5, ~0.5], [~0.5, ~0.5], [~0.5, ~0.5]]    loss = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=y)  # ~2.3    fe.backend.update_model(m, loss=loss, tape=tape)```
* **This method can be used with PyTorch models** : ```pythonm = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn="adam")x = torch.ones((3,1,28,28))  # (batch, channels, height, width)y = torch.tensor((1, 0, 1))pred = fe.backend.feed_forward(m, x)  # [[~0.5, ~0.5], [~0.5, ~0.5], [~0.5, ~0.5]]loss = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=y)  # ~2.3fe.backend.update_model(m, loss=loss)```

#### Args:

* **model** :  A neural network instance to update.
* **loss** :  A loss value to compute gradients from.
* **tape** :  A TensorFlow GradientTape which was recording when the `loss` was computed (iff using TensorFlow).
* **retain_graph** :  Whether to keep the model graph in memory (applicable only for PyTorch).

#### Raises:

* **ValueError** :  If `model` is an unacceptable data type.