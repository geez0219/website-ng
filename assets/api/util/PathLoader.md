## PathLoader
```python
PathLoader(root_path, batch=10, input_extension=None, recursive_search=True)
```


#### Args:

* **root_path** :  The path the the root directory containing files to be read
* **batch** :  The batch size to use when loading paths. Must be positive
* **input_extension** :  A file extension to limit what sorts of paths are returned
* **recursive_search** :  Whether to search within subdirectories for files