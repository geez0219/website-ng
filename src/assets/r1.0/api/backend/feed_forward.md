

### feed_forward
```python
feed_forward(model:Union[tensorflow.python.keras.engine.training.Model, torch.nn.modules.module.Module], x:Union[~Tensor, numpy.ndarray], training:bool=True) -> ~Tensor
```
Run a forward step on a given model.
* **This method can be used with TensorFlow models** : ```pythonm = fe.architecture.tensorflow.LeNet(classes=2)x = tf.ones((3,28,28,1))  # (batch, height, width, channels)b = fe.backend.feed_forward(m, x)  # [[~0.5, ~0.5], [~0.5, ~0.5], [~0.5, ~0.5]]```
* **This method can be used with PyTorch models** : ```pythonm = fe.architecture.pytorch.LeNet(classes=2)x = torch.ones((3,1,28,28))  # (batch, channels, height, width)b = fe.backend.feed_forward(m, x)  # [[~0.5, ~0.5], [~0.5, ~0.5], [~0.5, ~0.5]]```

#### Args:

* **model** :  A neural network to run the forward step through.
* **x** :  An input tensor for the `model`. This value will be auto-cast to either a tf.Tensor or torch.Tensor as        applicable for the `model`.
* **training** :  Whether this forward step is part of training or not. This may impact the behavior of `model` layers        such as dropout.

#### Returns:
    The result of `model(x)`.

#### Raises:

* **ValueError** :  If `model` is an unacceptable data type.