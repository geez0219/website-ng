## Traceability
```python
Traceability(
	save_path: str,
	extra_objects: Any=None
)
```
Automatically generate summary reports of the training.


#### Args:

* **save_path** :  Where to save the output files. Note that this will generate a new folder with the given name, into        which the report and corresponding graphics assets will be written.
* **extra_objects** :  Any extra objects which are not part of the Estimator, but which you want to capture in the        summary report. One example could be an extra pipeline which performs pre-processing.

#### Raises:

* **OSError** :  If graphviz is not installed.