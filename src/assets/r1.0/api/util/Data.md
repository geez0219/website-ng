## Data
```python
Data(*args, **kwds)
```
A class which contains prediction and batch data.    Data objects can be interacted with as if they are regular dictionaries. They are however, actually a combination of    two dictionaries, a dictionary for trace communication and a dictionary of prediction+batch data. In general, data    written into the trace dictionary will be logged by the system, whereas data in the pred+batch dictionary will not.    We therefore provide helper methods to write entries into `Data` which are intended or not intended for logging.    ```python
* **d = fe.util.Data({"a"** : 0, "b"1, "c"2})    a = d["a"]  # 0    d.write_with_log("d", 3)    d.write_without_log("e", 5)    d.write_with_log("a", 4)    a = d["a"]  # 4
* **r = d.read_logs(extra_keys={"c"})  # {"c"** : 2, "d"3, "a"4}    ```

#### Args:

* **batch_data** :  The batch data dictionary. In practice this is itself often a ChainMap containing separate            prediction and batch dictionaries.    

### read_logs
```python
read_logs(self, extra_keys:Set[str]) -> Dict[str, Any]
```
Read all values from the `Data` dictionary which were intended to be logged.

#### Args:

* **extra_keys** :  Any keys which should be logged, but which were not put into the `Data` dictionary via the                write_with_log function.

#### Returns:
            A dictionary of all of the keys and values to be logged.        

### write_with_log
```python
write_with_log(self, key:str, value:Any) -> None
```
Write a given `value` into the `Data` dictionary with the intent that it be logged.

#### Args:

* **key** :  The key to associate with the new entry.
* **value** :  The new entry to be written.        

### write_without_log
```python
write_without_log(self, key:str, value:Any) -> None
```
Write a given `value` into the `Data` dictionary with the intent that it not be logged.

#### Args:

* **key** :  The ey to associate with the new entry.
* **value** :  The new entry to be written.        