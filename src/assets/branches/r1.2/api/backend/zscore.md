## zscore<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/zscore.py/#L24-L69>View source on Github</a>
```python
zscore(
	data: ~Tensor,
	epsilon: float=1e-07
)
-> ~Tensor
```
Apply Zscore processing to a given tensor or array.

This method can be used with Numpy data:
```python
n = np.array([[0,1],[2,3]])
b = fe.backend.zscore(n)  # [[-1.34164079, -0.4472136 ],[0.4472136 , 1.34164079]]
```

This method can be used with TensorFlow tensors:
```python
t = tf.constant([[0,1],[2,3]])
b = fe.backend.zscore(t)  # [[-1.34164079, -0.4472136 ],[0.4472136 , 1.34164079]]
```

This method can be used with PyTorch tensors:
```python
p = torch.tensor([[0,1],[2,3]])
b = fe.backend.zscore(p)  # [[-1.34164079, -0.4472136 ],[0.4472136 , 1.34164079]]
```


<h3>Args:</h3>


* **data**: The input tensor or array. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    Data after substracting mean and divided by standard deviation.

</li></ul>

