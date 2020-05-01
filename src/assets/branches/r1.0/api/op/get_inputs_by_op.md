

### get_inputs_by_op
```python
get_inputs_by_op(op:op.op.Op, store:Mapping[str, Any]) -> Any
```
Retrieve the necessary input data from the data dictionary in order to run an `op`.

#### Args:

* **op** :  The op to run.
* **store** :  The system's data dictionary to draw inputs out of.

#### Returns:
    Input data to be fed to the `op` forward function.