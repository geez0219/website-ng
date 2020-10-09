## parse_modes<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/util/util.py/#L355-L383>View source on Github</a>
```python
parse_modes(
	modes: Set[str]
)
-> Set[str]
```
A function to determine which modes to run on based on a set of modes potentially containing blacklist values.

```python
m = fe.util.parse_modes({"train"})  # {"train"}
m = fe.util.parse_modes({"!train"})  # {"eval", "test", "infer"}
m = fe.util.parse_modes({"train", "eval"})  # {"train", "eval"}
m = fe.util.parse_modes({"!train", "!infer"})  # {"eval", "test"}
```


<h3>Args:</h3>


* **modes**: The desired modes to run on (possibly containing blacklisted modes). 

<h3>Raises:</h3>


* **AssertionError**: If invalid modes are detected, or if blacklisted modes and whitelisted modes are mixed.

<h3>Returns:</h3>

<ul class="return-block"><li>    The modes to run on (converted to a whitelist).

</li></ul>

