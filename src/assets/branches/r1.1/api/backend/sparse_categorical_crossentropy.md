## sparse_categorical_crossentropy<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/backend/sparse_categorical_crossentropy.py/#L25-L77>View source on Github</a>
```python
sparse_categorical_crossentropy(
	y_pred: ~Tensor,
	y_true: ~Tensor,
	from_logits: bool=False,
	average_loss: bool=True
)
-> ~Tensor
```
Compute sparse categorical crossentropy.

Note that if any of the `y_pred` values are exactly 0, this will result in a NaN output. If `from_logits` is
False, then each entry of `y_pred` should sum to 1. If they don't sum to 1 then tf and torch backends will
result in different numerical values.

This method can be used with TensorFlow tensors:
```python
true = tf.constant([[1], [0], [2]])
pred = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
b = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=true)  # 0.228
b = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=true, average_loss=False)  # [0.22, 0.11, 0.36]
```

This method can be used with PyTorch tensors:
```python
true = torch.tensor([[1], [0], [2]])
pred = torch.tensor([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
b = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=true)  # 0.228
b = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=true, average_loss=False)  # [0.22, 0.11, 0.36]
```


<h3>Args:</h3>


* **y_pred**: Prediction with a shape like (Batch, C). dtype: float32 or float16.

* **y_true**: Ground truth class labels with a shape like (Batch) or (Batch, 1). dtype: int.

* **from_logits**: Whether y_pred is from logits. If True, a softmax will be applied to the prediction.

* **average_loss**: Whether to average the element-wise loss. 

<h3>Raises:</h3>


* **AssertionError**: If `y_true` or `y_pred` are unacceptable data types.

<h3>Returns:</h3>

<ul class="return-block"><li>    The sparse categorical crossentropy between <code>y_pred</code> and <code>y_true</code>. A scalar if <code>average_loss</code> is True, else a
    tensor with the shape (Batch).

</li></ul>

