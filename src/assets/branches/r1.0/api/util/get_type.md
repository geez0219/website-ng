## get_type<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/util/util.py/#L286-L312>View source on Github</a>
```python
get_type(
	obj: Any
)
-> str
```
A function to try and infer the types of data within containers.

```python
x = fe.util.get_type(np.ones((10, 10), dtype='int32'))  # "int32"
x = fe.util.get_type(tf.ones((10, 10), dtype='float16'))  # "<dtype: 'float16'>"
x = fe.util.get_type(torch.ones((10, 10)).type(torch.float))  # "torch.float32"
x = fe.util.get_type([np.ones((10,10)) for i in range(4)])  # "List[float64]"
x = fe.util.get_type(27)  # "int"
```


<h3>Args:</h3>

* **obj** :  Data which may be wrapped in some kind of container.

<h3>Returns:</h3>
    A string representation of the data type of the `obj`.

