## mean_squared_error<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/backend/mean_squared_error.py/#L25-L69>View source on Github</a>
```python
mean_squared_error(
	y_true: ~Tensor,
	y_pred: ~Tensor
)
-> ~Tensor
```
Calculate mean squared error between two tensors.

This method can be used with TensorFlow tensors:
```python
true = tf.constant([[0,1,0,0], [0,0,0,1], [0,0,1,0], [1,0,0,0]])
pred = tf.constant([[0.1,0.9,0.05,0.05], [0.1,0.2,0.0,0.7], [0.0,0.15,0.8,0.05], [1.0,0.0,0.0,0.0]])
b = fe.backend.mean_squared_error(y_pred=pred, y_true=true)  # [0.0063, 0.035, 0.016, 0.0]
true = tf.constant([[1], [3], [2], [0]])
pred = tf.constant([[2.0], [0.0], [2.0], [1.0]])
b = fe.backend.mean_squared_error(y_pred=pred, y_true=true)  # [1.0, 9.0, 0.0, 1.0]
```

This method can be used with PyTorch tensors:
```python
true = torch.tensor([[0,1,0,0], [0,0,0,1], [0,0,1,0], [1,0,0,0]])
pred = torch.tensor([[0.1,0.9,0.05,0.05], [0.1,0.2,0.0,0.7], [0.0,0.15,0.8,0.05], [1.0,0.0,0.0,0.0]])
b = fe.backend.mean_squared_error(y_pred=pred, y_true=true)  # [0.0063, 0.035, 0.016, 0.0]
true = tf.constant([[1], [3], [2], [0]])
pred = tf.constant([[2.0], [0.0], [2.0], [1.0]])
b = fe.backend.mean_squared_error(y_pred=pred, y_true=true)  # [1.0, 9.0, 0.0, 1.0]
```


<h3>Args:</h3>


* **y_true**: Ground truth class labels with a shape like (batch) or (batch, n_classes). dtype: int, float16, float32.

* **y_pred**: Prediction score for each class, with a shape like y_true. dtype: float32 or float16. 

<h3>Raises:</h3>


* **AssertionError**: If `y_true` and `y_pred` have mismatched shapes or disparate types.

* **ValueError**: If `y_pred` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    The MSE between <code>y_true</code> and <code>y_pred</code>

</li></ul>

