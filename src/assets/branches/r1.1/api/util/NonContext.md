## NonContext<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/util/util.py/#L187-L203>View source on Github</a>
```python
NonContext(
	*args, **kwargs
)
```
A class which is used to make nothing unusual happen.

This class is intentionally not @traceable.

```python
a = 5
with fe.util.NonContext():
    a = a + 37
print(a)  # 42
```


