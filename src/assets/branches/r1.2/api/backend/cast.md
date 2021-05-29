## cast<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/backend/cast.py/#L26-L75>View source on Github</a>
```python
cast(
	data: Union[Collection, ~Tensor],
	dtype: str
)
-> Union[Collection, ~Tensor]
```
Cast the data to a specific data type recursively.

This method can be used with Numpy data:
 ```python
 data = {"x": np.ones((10,15)), "y":[np.ones((4)), np.ones((5, 3))], "z":{"key":np.ones((2,2))}}
 fe.backend.to_type(data)
 # {'x': dtype('float64'), 'y': [dtype('float64'), dtype('float64')], 'z': {'key': dtype('float64')}}
 data = fe.backend.cast(data, "float16")
 fe.backend.to_type(data)
 # {'x': dtype('float16'), 'y': [dtype('float16'), dtype('float16')], 'z': {'key': dtype('float16')}}
 ```

 This method can be used with TensorFlow tensors:
 ```python
 data = {"x": tf.ones((10,15)), "y":[tf.ones((4)), tf.ones((5, 3))], "z":{"key":tf.ones((2,2))}}
 fe.backend.to_type(data) # {'x': tf.float32, 'y': [tf.float32, tf.float32], 'z': {'key': tf.float32}}
 data = fe.backend.cast(data, "uint8")
 fe.backend.to_type(data) # {'x': tf.uint8, 'y': [tf.uint8, tf.uint8], 'z': {'key': tf.uint8}}
 ```

 This method can be used with PyTorch tensors:
 ```python
 data = {"x": torch.ones((10,15)), "y":[torch.ones((4)), torch.ones((5, 3))], "z":{"key":torch.ones((2,2))}}
 fe.backend.to_type(data) # {'x': torch.float32, 'y': [torch.float32, torch.float32], 'z': {'key': torch.float32}}
 data = fe.backend.cast(data, "float64")
 fe.backend.to_type(data) # {'x': torch.float64, 'y': [torch.float64, torch.float64], 'z': {'key': torch.float64}}
 ```



<h3>Args:</h3>


* **data**: A tensor or possibly nested collection of tensors.

* **dtype**: Target data type, can be one of following: uint8, int8, int16, int32, int64, float16, float32, float64. 

<h3>Returns:</h3>

<ul class="return-block"><li>     A collection with the same structure as <code>data</code> with target data type.
 </li></ul>

