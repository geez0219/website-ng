## to_list<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/util/util.py/#L130-L156>View source on Github</a>
```python
to_list(
	data: Any
)
-> List[Any]
```
Convert data to a list. A single None value will be converted to the empty list.

```python
x = fe.util.to_list(None)  # []
x = fe.util.to_list([None])  # [None]
x = fe.util.to_list(7)  # [7]
x = fe.util.to_list([7, 8])  # [7,8]
x = fe.util.to_list({7})  # [7]
x = fe.util.to_list((7))  # [7]
x = fe.util.to_list({'a': 7})  # [{'a': 7}]
```


<h3>Args:</h3>


* **data**: Input data, within or without a python container. 

<h3>Returns:</h3>

<ul class="return-block"><li>    The input <code>data</code> but inside a list instead of whatever other container type used to hold it.</li></ul>

