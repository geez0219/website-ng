## Repeat
```python
Repeat(*args, **kwargs)
```
Repeat a NumpyOp several times in a row.


#### Args:

* **op** :  A NumpyOp to be run one or more times in a row.
* **repeat** :  How many times to repeat the `op`. This can also be a function return, in which case the function input        names will be matched to keys in the data dictionary, and the `op` will be repeated until the function        evaluates to False. The function evaluation will happen at the end of a forward call, so the `op` will        always be evaluated at least once.

#### Raises:

* **ValueError** :  If `repeat` or `op` are invalid.