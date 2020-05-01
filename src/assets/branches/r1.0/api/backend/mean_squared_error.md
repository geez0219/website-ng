

### mean_squared_error
```python
mean_squared_error(y_true:~Tensor, y_pred:~Tensor) -> ~Tensor
```
Calculate mean squared error between two tensors.
* **This method can be used with TensorFlow tensors** : ```pythontrue = tf.constant([[0,1,0,0], [0,0,0,1], [0,0,1,0], [1,0,0,0]])pred = tf.constant([[0.1,0.9,0.05,0.05], [0.1,0.2,0.0,0.7], [0.0,0.15,0.8,0.05], [1.0,0.0,0.0,0.0]])b = fe.backend.mean_squared_error(y_pred=pred, y_true=true)  # [0.0063, 0.035, 0.016, 0.0]true = tf.constant([[1], [3], [2], [0]])pred = tf.constant([[2.0], [0.0], [2.0], [1.0]])b = fe.backend.mean_squared_error(y_pred=pred, y_true=true)  # [1.0, 9.0, 0.0, 1.0]```
* **This method can be used with PyTorch tensors** : ```pythontrue = torch.tensor([[0,1,0,0], [0,0,0,1], [0,0,1,0], [1,0,0,0]])pred = torch.tensor([[0.1,0.9,0.05,0.05], [0.1,0.2,0.0,0.7], [0.0,0.15,0.8,0.05], [1.0,0.0,0.0,0.0]])b = fe.backend.mean_squared_error(y_pred=pred, y_true=true)  # [0.0063, 0.035, 0.016, 0.0]true = tf.constant([[1], [3], [2], [0]])pred = tf.constant([[2.0], [0.0], [2.0], [1.0]])b = fe.backend.mean_squared_error(y_pred=pred, y_true=true)  # [1.0, 9.0, 0.0, 1.0]```

#### Args:

* **y_true** :  Ground truth class labels with a shape like (batch) or (batch, n_classes). dtype int or float32.
* **y_pred** :  Prediction score for each class, with a shape like y_true. dtype float32.

#### Returns:
    The MSE between `y_true` and `y_pred`

#### Raises:

* **AssertionError** :  If `y_true` and `y_pred` have mismatched shapes or disparate types.
* **ValueError** :  If `y_pred` is an unacceptable data type.