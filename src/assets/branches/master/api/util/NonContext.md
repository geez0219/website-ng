## NonContext
```python
NonContext()
```
A class which is used to make nothing unusual happen.

This class is intentionally not @traceable.

```python
a = 5
with fe.util.NonContext():
    a = a + 37
print(a)  # 42
```
