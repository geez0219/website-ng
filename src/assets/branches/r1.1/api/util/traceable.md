## traceable<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/util/traceability_util.py/#L1017-L1084>View source on Github</a>
```python
traceable(
	whitelist: Union[str, Tuple[str]]=(),
	blacklist: Union[str, Tuple[str]]=()
)
-> Callable
```
A decorator to be placed on classes in order to make them traceable and to enable a deep restore.

Decorated classes will gain the .fe_summary() and .fe_state() methods.


<h3>Args:</h3>

* **whitelist** :  Arguments which should be included in a deep restore of the decorated class.
* **blacklist** :  Arguments which should be excluded from a deep restore of the decorated class.

<h3>Returns:</h3>
    The decorated class.

