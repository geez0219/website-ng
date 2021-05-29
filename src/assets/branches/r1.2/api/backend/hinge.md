## hinge<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/hinge.py/#L27-L58>View source on Github</a>
```python
hinge(
	y_true: ~Tensor,
	y_pred: ~Tensor
)
-> ~Tensor
```
Calculate the hinge loss between two tensors.

This method can be used with TensorFlow tensors:
```python
true = tf.constant([[-1,1,1,-1], [1,1,1,1], [-1,-1,1,-1], [1,-1,-1,-1]])
pred = tf.constant([[0.1,0.9,0.05,0.05], [0.1,-0.2,0.0,-0.7], [0.0,0.15,0.8,0.05], [1.0,-1.0,-1.0,-1.0]])
b = fe.backend.hinge(y_pred=pred, y_true=true)  # [0.8  1.2  0.85 0.  ]
```

This method can be used with PyTorch tensors:
```python
true = torch.tensor([[-1,1,1,-1], [1,1,1,1], [-1,-1,1,-1], [1,-1,-1,-1]])
pred = torch.tensor([[0.1,0.9,0.05,0.05], [0.1,-0.2,0.0,-0.7], [0.0,0.15,0.8,0.05], [1.0,-1.0,-1.0,-1.0]])
b = fe.backend.hinge(y_pred=pred, y_true=true)  # [0.8  1.2  0.85 0.  ]
```


<h3>Args:</h3>


* **y_true**: Ground truth class labels which should take values of 1 or -1.

* **y_pred**: Prediction score for each class, with a shape like y_true. dtype: float32 or float16. 

<h3>Raises:</h3>


* **AssertionError**: If `y_true` and `y_pred` have mismatched shapes or disparate types.

* **ValueError**: If `y_pred` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    The hinge loss between <code>y_true</code> and <code>y_pred</code>

</li></ul>

