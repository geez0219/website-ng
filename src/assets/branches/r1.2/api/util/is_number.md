## is_number<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/util/util.py/#L567-L585>View source on Github</a>
```python
is_number(
	arg: str
)
-> bool
```
Check if a given string can be converted into a number.

```python
x = fe.util.is_number("13.7")  # True
x = fe.util.is_number("ae13.7")  # False
```


<h3>Args:</h3>


* **arg**: A potentially numeric input string. 

<h3>Returns:</h3>

<ul class="return-block"><li>    True iff <code>arg</code> represents a number.</li></ul>

