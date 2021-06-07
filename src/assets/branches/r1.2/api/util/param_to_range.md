## param_to_range<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/util/util.py/#L188-L213>View source on Github</a>
```python
param_to_range(
	data: Union[int, float, Tuple[int, int], Tuple[float, float]]
)
-> Union[Tuple[int, int], Tuple[float, float]]
```
Convert a single int or float value to a tuple signifying a range.

```python
x = fe.util.param_to_tuple(7)  # (-7, 7)
x = fe.util.param_to_tuple([7, 8])  # (7,8))
x = fe.util.param_to_tuple((3.1, 4.3))  # (3.1, 4.3)
x = fe.util.to_set((-3.2))  # (-3.2, 3.2)
```


<h3>Args:</h3>


* **data**: Input data. 

<h3>Returns:</h3>

<ul class="return-block"><li>    The input <code>data</code> but in tuple form for a range.</li></ul>

