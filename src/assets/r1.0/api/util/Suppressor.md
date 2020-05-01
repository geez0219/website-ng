## Suppressor
```python
Suppressor()
```
A class which can be used to silence output of function calls.    ```python
* **x = lambda** :  print("hello")    x()  # "hello"
* **with fe.util.Suppressor()** :         x()  #    x()  # "hello"    ```    

### write
```python
write(self, dummy:str) -> None
```
A function which is invoked during print calls.

#### Args:

* **dummy** :  The string which wanted to be printed.        