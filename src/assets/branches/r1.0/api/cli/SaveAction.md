## SaveAction
```python
SaveAction(option_strings:Sequence[str], dest:str, nargs:Union[int, str, NoneType]='?', **kwargs:Dict[str, Any]) -> None
```
A customized save action for use with argparse.    A custom save action which is used to populate a secondary variable inside of an exclusive group. Used if this file    is invoked directly during argument parsing.

#### Args:

* **option_strings** :  A list of command-line option strings which should be associated with this action.
* **dest** :  The name of the attribute to hold the created object(s).
* **nargs** :  The number of command line arguments to be consumed.
 **kwargs :  Pass-through keyword arguments.    