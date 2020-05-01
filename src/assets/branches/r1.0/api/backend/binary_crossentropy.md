

### binary_crossentropy
```python
binary_crossentropy(y_pred:~Tensor, y_true:~Tensor, from_logits:bool=False, average_loss:bool=True) -> ~Tensor
```
Compute binary crossentropy.This method is applicable when there are only two label classes (zero and one). There should be a single floatingpoint prediction per example.
* **This method can be used with TensorFlow tensors** : ```pythontrue = tf.constant([[1], [0], [1], [0]])pred = tf.constant([[0.9], [0.3], [0.8], [0.1]])b = fe.backend.binary_crossentropy(y_pred=pred, y_true=true)  # 0.197b = fe.backend.binary_crossentropy(y_pred=pred, y_true=true, average_loss=False)  # [0.105, 0.356, 0.223, 0.105]```
* **This method can be used with PyTorch tensors** : ```pythontrue = torch.tensor([[1], [0], [1], [0]])pred = torch.tensor([[0.9], [0.3], [0.8], [0.1]])b = fe.backend.binary_crossentropy(y_pred=pred, y_true=true)  # 0.197b = fe.backend.binary_crossentropy(y_pred=pred, y_true=true, average_loss=False)  # [0.105, 0.356, 0.223, 0.105]```

#### Args:

* **y_pred** :  Prediction with a shape like (batch, ...). dtype float32.
* **y_true** :  Ground truth class labels with the same shape as `y_pred`. dtype int or float32.
* **from_logits** :  Whether y_pred is from logits. If True, a sigmoid will be applied to the prediction.
* **average_loss** :  Whether to average the element-wise loss.

#### Returns:
    The binary crossentropy between `y_pred` and `y_true`. A scalar if `average_loss` is True, else a tensor with    the same shape as `y_true`.

#### Raises:

* **AssertionError** :  If `y_true` or `y_pred` are unacceptable data types.