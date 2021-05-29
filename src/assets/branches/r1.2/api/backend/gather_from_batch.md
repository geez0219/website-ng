## gather_from_batch<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/gather_from_batch.py/#L28-L82>View source on Github</a>
```python
gather_from_batch(
	tensor: ~Tensor,
	indices: ~Tensor
)
-> ~Tensor
```
Gather specific indices from a batch of data.

This method can be useful if you need to compute gradients based on a specific subset of a tensor's output values.
The `indices` will automatically be cast to the correct type (tf, torch, np) based on the type of the `tensor`.

This method can be used with Numpy data:
```python
ind = np.array([1, 0, 1])
n = np.array([[0, 1], [2, 3], [4, 5]])
b = fe.backend.gather_from_batch(n, ind)  # [1, 2, 5]
n = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
b = fe.backend.gather_from_batch(n, ind)  # [[2, 3], [4, 5], [10, 11]]
```

This method can be used with TensorFlow tensors:
```python
ind = tf.constant([1, 0, 1])
t = tf.constant([[0, 1], [2, 3], [4, 5]])
b = fe.backend.gather_from_batch(t, ind)  # [1, 2, 5]
t = tf.constant([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
b = fe.backend.gather_from_batch(t, ind)  # [[2, 3], [4, 5], [10, 11]]
```

This method can be used with PyTorch tensors:
```python
ind = torch.tensor([1, 0, 1])
p = torch.tensor([[0, 1], [2, 3], [4, 5]])
b = fe.backend.gather_from_batch(p, ind)  # [1, 2, 5]
p = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
b = fe.backend.gather_from_batch(p, ind)  # [[2, 3], [4, 5], [10, 11]]
```


<h3>Args:</h3>


* **tensor**: A tensor of shape (batch, d1, ..., dn).

* **indices**: A tensor of shape (batch, ) or (batch, 1) indicating which indices should be selected. 

<h3>Raises:</h3>


* **ValueError**: If `tensor` is an unacceptable data type.

<h3>Returns:</h3>

<ul class="return-block"><li>    A tensor of shape (batch, d2, ..., dn) containing the elements from <code>tensor</code> at the given <code>indices</code>.

</li></ul>

