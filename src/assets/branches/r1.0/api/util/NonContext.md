## NonContext<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/util/util.py/#L140-L154>View source on Github</a>
```python
NonContext()
```
A class which is used to make nothing unusual happen.

```python
a = 5
with fe.util.NonContext():
    a = a + 37
print(a)  # 42
```




