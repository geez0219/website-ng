## ReflectionPadding2D<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/layers/tensorflow/reflection_padding_2d.py/#L21-L53>View source on Github</a>
```python
ReflectionPadding2D(
	*args, **kwargs
)
```
A layer for performing Reflection Padding on 2D arrays.

This layer assumes that you are using the a tensor shaped like (Batch, Height, Width, Channels).
The implementation here is borrowed from https://stackoverflow.com/questions/50677544/reflection-padding-conv2d.

```python
x = tf.reshape(tf.convert_to_tensor(list(range(9))), (1,3,3,1))  # ~ [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
m = fe.layers.tensorflow.ReflectionPadding2D((1, 1))
y = m(x)  # ~ [[4, 3, 4, 5, 4], [1, 0, 1, 2, 1], [4, 3, 4, 5, 4], [7, 6, 7, 8, 7], [4, 3, 4, 5, 4]]
m = fe.layers.tensorflow.ReflectionPadding2D((1, 0))
y = m(x)  # ~ [[1, 0, 1, 2, 1], [4, 3, 4, 5, 4], [7, 6, 7, 8, 7]]
```


<h3>Args:</h3>

* **padding** :  padding size (Width, Height). The padding size must be less than the size of the corresponding        dimension in the input tensor.

### compute_output_shape<span class="tag">method of ReflectionPadding2D</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/layers/tensorflow/reflection_padding_2d.py/#L47-L49>View source on Github</a>
```python
compute_output_shape(
	self,
	s: Tuple[int, int, int, int]
)
-> Tuple[int, int, int, int]
```
If you are using "channels_last" configuration



