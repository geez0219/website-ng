## get_shape<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/util/util.py/#L315-L352>View source on Github</a>
```python
get_shape(
	obj: Any
)
-> List[Union[int, NoneType]]
```
A function to find the shapes of an object or sequence of objects.

Lists or Tuples will assume that the zeroth dimension is ragged (shape==None). If entries in the list have
mismatched ranks, then only the list dimension will be considered as part of the shape. If all ranks are equal, an
attempt will be made to determine which of the interior dimensions are ragged.

```python
x = fe.util.get_shape(np.ones((12,22,11)))  # [12, 22, 11]
x = fe.util.get_shape([np.ones((12,22,11)), np.ones((18, 5))])  # [None]
x = fe.util.get_shape([np.ones((12,22,11)), np.ones((18, 5, 4))])  # [None, None, None, None]
x = fe.util.get_shape([np.ones((12,22,11)), np.ones((12, 22, 4))])  # [None, 12, 22, None]
x = fe.util.get_shape({"a": np.ones((12,22,11))})  # []
```


<h3>Args:</h3>

* **obj** :  Data to infer the shape of.

<h3>Returns:</h3>
    A list representing the shape of the data.

