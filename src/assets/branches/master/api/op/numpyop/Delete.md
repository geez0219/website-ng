## Delete
```python
Delete(*args, **kwargs)
```
Delete key(s) and their associated values from the data dictionary.

The system has special logic to detect instances of this Op and delete its `inputs` from the data dictionary.


#### Args:

* **keys** :  Existing key(s) to be deleted from the data dictionary.