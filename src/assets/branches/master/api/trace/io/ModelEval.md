## ModelEval
```python
ModelEval(*args, **kwargs)
```
Automate testing and report generation.

This trace will examine all input TestCases in the test mode and generate a PDF report and a JSON test result.



#### Args:

* **test_cases** :  TestCase object or list of TestCase objects.
* **save_path** :  Where to save the output directory.
* **test_title** :  Title of the test.
* **data_id** :  Data instance ID key. If provided, then per-instance test will return failure instance ID.

### check_pdf_dependency
```python
check_pdf_dependency() -> None
```
Check dependency of PDF-generating packages.


#### Raises:

* **OSError** :  Some required package has not been installed.