## Loss
```python
Loss(inputs=None, outputs=None, mode=None)
```
A base class for loss operations. It can be used directly to perform value pass-through (see the adversarialtraining showcase for an example of when this is useful)

### validate_loss_inputs
```python
validate_loss_inputs(inputs, *args)
```
A method to ensure that either the inputs array or individual input arguments are specified, but not both

#### Args:

* **inputs** :  None or a tuple/list of arguments
 *args :  a tuple of arguments or Nones

#### Returns:
            either 'inputs' or the args tuple depending on which is populated        