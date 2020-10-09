## Suppressor<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/util/util.py/#L157-L186>View source on Github</a>
```python
Suppressor(
	*args, **kwargs
)
```
A class which can be used to silence output of function calls.

```python
x = lambda: print("hello")
x()  # "hello"
with fe.util.Suppressor():
    x()  #
x()  # "hello"
```


---

### write<span class="tag">method of Suppressor</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/util/util.py/#L180-L186>View source on Github</a>
```python
write(
	self,
	dummy: str
)
-> None
```
A function which is invoked during print calls.


<h4>Args:</h4>


* **dummy**: The string which wanted to be printed.

