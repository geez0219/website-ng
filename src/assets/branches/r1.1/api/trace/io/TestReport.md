## TestReport<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/io/test_report.py/#L86-L517>View source on Github</a>
```python
TestReport(
	test_cases: Union[fastestimator.trace.io.test_report.TestCase, List[fastestimator.trace.io.test_report.TestCase]],
	save_path: str,
	test_title: Union[str, NoneType]=None,
	data_id: str=None
)
-> None
```
Automate testing and report generation.

This trace will evaluate all its `test_cases` during test mode and generate a PDF report and a JSON test result.


<h3>Args:</h3>


* **test_cases**: The test(s) to be run.

* **save_path**: Where to save the outputs.

* **test_title**: The title of the test, or None to use the experiment name.

* **data_id**: Data instance ID key. If provided, then per-instances test will include failing instance IDs.

---

### check_pdf_dependency<span class="tag">method of TestReport</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/io/test_report.py/#L490-L502>View source on Github</a>
```python
check_pdf_dependency()
-> None
```
Check dependency of PDF-generating packages.


<h4>Raises:</h4>


* **OSError**: Some required package has not been installed.

---

### sanitize_value<span class="tag">method of TestReport</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/io/test_report.py/#L504-L517>View source on Github</a>
```python
sanitize_value(
	value: Union[int, float]
)
-> str
```
Sanitize input value for a better report display.


<h4>Args:</h4>


* **value**: Value to be sanitized. 

<h4>Returns:</h4>

<ul class="return-block"><li>    Sanitized string of <code>value</code>.</li></ul>

