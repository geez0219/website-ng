

### forward_numpyop
```python
forward_numpyop(ops:List[op.numpyop.numpyop.NumpyOp], data:MutableMapping[str, Any], mode:str) -> None
```
Call the forward function for list of NumpyOps, and modify the data dictionary in place.

#### Args:

* **ops** :  A list of NumpyOps to execute.
* **data** :  The data dictionary.
* **mode** :  The current execution mode ("train", "eval", "test", or "infer").