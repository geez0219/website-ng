## to_set<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/util/util.py/#L112-L137>View source on Github</a>
```python
to_set(
	data: Any
)
-> Set[Any]
```
Convert data to a set. A single None value will be converted to the empty set.

```python
x = fe.util.to_set(None)  # {}
x = fe.util.to_set([None])  # {None}
x = fe.util.to_set(7)  # {7}
x = fe.util.to_set([7, 8])  # {7,8}
x = fe.util.to_set({7})  # {7}
x = fe.util.to_set((7))  # {7}
```


<h3>Args:</h3>

* **data** :  Input data, within or without a python container. The `data` must be hashable.

<h3>Returns:</h3>
    The input `data` but inside a set instead of whatever other container type used to hold it.

