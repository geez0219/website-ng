

### categorical_crossentropy
```python
categorical_crossentropy(y_pred:~Tensor, y_true:~Tensor, from_logits:bool=False, average_loss:bool=True) -> ~Tensor
```
Compute categorical crossentropy.Note that if any of the `y_pred` values are exactly 0, this will result in a NaN output. If `from_logits` isFalse, then each entry of `y_pred` should sum to 1. If they don't sum to 1 then tf and torch backends willresult in different numerical values.
* **This method can be used with TensorFlow tensors** : ```pythontrue = tf.constant([[0, 1, 0], [1, 0, 0], [0, 0, 1]])pred = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])b = fe.backend.categorical_crossentropy(y_pred=pred, y_true=true)  # 0.228b = fe.backend.categorical_crossentropy(y_pred=pred, y_true=true, average_loss=False)  # [0.223, 0.105, 0.356]```
* **This method can be used with PyTorch tensors** : ```pythontrue = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])pred = torch.tensor([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])b = fe.backend.categorical_crossentropy(y_pred=pred, y_true=true)  # 0.228b = fe.backend.categorical_crossentropy(y_pred=pred, y_true=true, average_loss=False)  # [0.223, 0.105, 0.356]```

#### Args:

* **y_pred** :  Prediction with a shape like (Batch, C). dtype float32.
* **y_true** :  Ground truth class labels with a shape like `y_pred`. dtype int or float32.
* **from_logits** :  Whether y_pred is from logits. If True, a sigmoid will be applied to the prediction.
* **average_loss** :  Whether to average the element-wise loss.

#### Returns:
    The categorical crossentropy between `y_pred` and `y_true`. A scalar if `average_loss` is True, else a    tensor with the shape (Batch).

#### Raises:

* **AssertionError** :  If `y_true` or `y_pred` are unacceptable data types.