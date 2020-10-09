## pad_data<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/util/util.py/#L507-L529>View source on Github</a>
```python
pad_data(
	data: numpy.ndarray,
	target_shape: Tuple[int, ...],
	pad_value: Union[float, int]
)
-> numpy.ndarray
```
Pad `data` by appending `pad_value`s along it's dimensions until the `target_shape` is reached. All entris of
target_shape should be larger than the data.shape, and have the same rank.

```python
x = np.ones((1,2))
x = fe.util.pad_data(x, target_shape=(3, 3), pad_value = -2)  # [[1, 1, -2], [-2, -2, -2], [-2, -2, -2]]
x = fe.util.pad_data(x, target_shape=(3, 3, 3), pad_value = -2) # error
x = fe.util.pad_data(x, target_shape=(4, 1), pad_value = -2) # error
```


<h3>Args:</h3>


* **data**: The data to be padded.

* **target_shape**: The desired shape for `data`. Should have the same rank as `data`, with each dimension being >= the size of the `data` dimension.

* **pad_value**: The value to insert into `data` if padding is required to achieve the `target_shape`. 

<h3>Returns:</h3>

<ul class="return-block"><li>    The <code>data</code>, padded to the <code>target_shape</code>.</li></ul>

