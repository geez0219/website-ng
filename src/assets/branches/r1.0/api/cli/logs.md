

### logs
```python
logs(args:Dict[str, Any], unknown:List[str]) -> None
```
A method to invoke the FE logging function using CLI-provided arguments.

#### Args:

* **args** :  The arguments to be fed to the parse_log_dir() method.
* **unknown** :  Any cli arguments not matching known inputs for the parse_log_dir() method.

#### Raises:

* **SystemExit** :  If `unknown` arguments were provided by the user.