## TestReport
```python
TestReport(*args, **kwargs)
```
Automate testing and report generation.

This trace will evaluate all its `test_cases` during test mode and generate a PDF report and a JSON test result.


#### Args:

* **test_cases** :  The test(s) to be run.
* **save_path** :  Where to save the outputs.
* **test_title** :  The title of the test, or None to use the experiment name.
* **data_id** :  Data instance ID key. If provided, then per-instances test will include failing instance IDs.

### check_pdf_dependency
```python
check_pdf_dependency() -> None
```
Check dependency of PDF-generating packages.


#### Raises:

* **OSError** :  Some required package has not been installed.