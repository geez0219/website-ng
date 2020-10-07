## strip_suffix<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/util/util.py/#L240-L260>View source on Github</a>
```python
strip_suffix(
	target: Union[str, NoneType],
	suffix: Union[str, NoneType]
)
-> Union[str, NoneType]
```
Remove the given `suffix` from the `target` if it is present there.

```python
x = fe.util.strip_suffix("astring.json", ".json")  # "astring"
x = fe.util.strip_suffix("astring.json", ".yson")  # "astring.json"
```


<h3>Args:</h3>

* **target** :  A string to be formatted.
* **suffix** :  A string to be removed from `target`.

<h3>Returns:</h3>
    The formatted version of `target`.

