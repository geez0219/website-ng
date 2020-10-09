## binary_crossentropy<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/backend/binary_crossentropy.py/#L25-L80>View source on Github</a>
```python
binary_crossentropy(
	y_pred: ~Tensor,
	y_true: ~Tensor,
	from_logits: bool=False,
	average_loss: bool=True
)
-> ~Tensor
```
Compute binary crossentropy.

This method is applicable when there are only two label classes (zero and one). There should be a single floating
point prediction per example.

This method can be used with TensorFlow tensors:
```python
true = tf.constant([[1], [0], [1], [0]])
pred = tf.constant([[0.9], [0.3], [0.8], [0.1]])
b = fe.backend.binary_crossentropy(y_pred=pred, y_true=true)  # 0.197
b = fe.backend.binary_crossentropy(y_pred=pred, y_true=true, average_loss=False)  # [0.105, 0.356, 0.223, 0.105]
```

This method can be used with PyTorch tensors:
```python
true = torch.tensor([[1], [0], [1], [0]])
pred = torch.tensor([[0.9], [0.3], [0.8], [0.1]])
b = fe.backend.binary_crossentropy(y_pred=pred, y_true=true)  # 0.197
b = fe.backend.binary_crossentropy(y_pred=pred, y_true=true, average_loss=False)  # [0.105, 0.356, 0.223, 0.105]
```


<h3>Args:</h3>


* **y_pred**: Prediction with a shape like (batch, ...). dtype: float32 or float16.

* **y_true**: Ground truth class labels with the same shape as `y_pred`. dtype: int or float32 or float16.

* **from_logits**: Whether y_pred is from logits. If True, a sigmoid will be applied to the prediction.

* **average_loss**: Whether to average the element-wise loss. 

<h3>Raises:</h3>


* **AssertionError**: If `y_true` or `y_pred` are unacceptable data types.

<h3>Returns:</h3>

<ul class="return-block"><li>    The binary crossentropy between <code>y_pred</code> and <code>y_true</code>. A scalar if <code>average_loss</code> is True, else a tensor with
    the same shape as <code>y_true</code>.

</li></ul>

