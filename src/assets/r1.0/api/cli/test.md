

### test
```python
test(args:Dict[str, Any], unknown:Union[List[str], NoneType]) -> None
```
Load an Estimator from a file and invoke its .test() method.

#### Args:

* **args** :  A dictionary containing location of the FE file under the 'entry_point' key, as well as an optional        'hyperparameters_json' key if the user is storing their parameters in a file.
* **unknown** :  The remainder of the command line arguments to be passed along to the get_estimator() method.