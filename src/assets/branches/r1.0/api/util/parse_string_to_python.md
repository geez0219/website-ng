## parse_string_to_python<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/util/util.py/#L57-L80>View source on Github</a>
```python
parse_string_to_python(
	val: str
)
-> Any
```
Convert a string into a python object.

```python
x = fe.util.parse_string_to_python("5")  # 5
x = fe.util.parse_string_to_python("[5, 4, 0.3]")  # [5, 4, 0.3]
x = fe.util.parse_string_to_python("{'a':5, 'b':7}")  # {'a':5, 'b':7}
```


<h3>Args:</h3>


* **val**: An input string. 

<h3>Returns:</h3>

<ul class="return-block"><li>    A python object version of the input string.</li></ul>

