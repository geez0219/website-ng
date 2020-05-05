

### watch
```python
watch(tensor:~Tensor, tape:Union[tensorflow.python.eager.backprop.GradientTape, NoneType]=None) -> ~Tensor
```
Monitor the given `tensor` for later gradient computations.
* **This method can be used with TensorFlow tensors** : ```pythonx = tf.ones((3,28,28,1))
* **with tf.GradientTape(persistent=True) as tape** :     x = fe.backend.watch(x, tape=tape)```
* **This method can be used with PyTorch tensors** : ```pythonx = torch.ones((3,1,28,28))  # x.requires_grad == Falsex = fe.backend.watch(x)  # x.requires_grad == True```

#### Args:

* **tensor** :  The tensor to be monitored.
* **tape** :  A TensorFlow GradientTape which will be used to record gradients (iff using TensorFlow for the backend).

#### Returns:
    The `tensor` or a copy of the `tensor` which is being tracked for gradient computations. This value is only    needed if using PyTorch as the backend.

#### Raises:

* **ValueError** :  If `tensor` is an unacceptable data type.