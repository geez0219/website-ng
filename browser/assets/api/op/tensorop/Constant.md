## Constant
```python
Constant(shape_like, constant=0, outputs=None, mode=None)
```
 A class to introduce a constant into the state dictionary

#### Args:

* **shape_like (str)** :  Key of a variable whose shape will be matched
* **constant (int, float, func)** :  A constant or function defining the value to be output. If a function is provided,                                    the desired shape will be passed as an argument.
* **outputs (str)** :  The name of the output value
* **mode (str)** :  Which mode to run in ('train', 'eval', None)    

### forward
```python
forward(self, data, state)
```
 This class is to be used to compute a constant of a particular shape

#### Args:

* **data** :  input data to define the constant's shape
* **state** :  Information about the current execution context.

#### Returns:
            a constant tensor of the same shape as the input        