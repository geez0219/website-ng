## strip_prefix<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/util/util.py/#L263-L283>View source on Github</a>
```python
strip_prefix(
	target: Union[str, NoneType],
	prefix: Union[str, NoneType]
)
-> Union[str, NoneType]
```
Remove the given `prefix` from the `target` if it is present there.

```python
x = fe.util.strip_prefix("astring.json", "ast")  # "ring.json"
x = fe.util.strip_prefix("astring.json", "asa")  # "astring.json"
```


<h3>Args:</h3>


* **target**: A string to be formatted.

* **prefix**: A string to be removed from `target`. 

<h3>Returns:</h3>

<ul class="return-block"><li>    The formatted version of <code>target</code>.</li></ul>

