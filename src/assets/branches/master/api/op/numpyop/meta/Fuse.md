## Fuse
```python
Fuse(*args, **kwargs)
```
Run a sequence of NumpyOps as a single Op.


#### Args:

* **ops** :  A sequence of NumpyOps to run. They must all share the same mode. It also doesn't support scheduled ops at        the moment, though the Fuse itself may be scheduled.

#### Raises:

* **ValueError** :  If `repeat` or `ops` are invalid.