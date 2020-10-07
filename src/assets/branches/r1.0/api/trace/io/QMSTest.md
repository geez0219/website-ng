## QMSTest<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/trace/io/qms.py/#L30-L101>View source on Github</a>
```python
QMSTest(
	test_descriptions: Union[str, List[str]],
	test_criterias: Union[List[Callable], Callable],
	test_title: str='QMSTest',
	json_output: str='',
	doc_output: str=''
)
-> None
```
Automate QMS testing and report generation.


<h3>Args:</h3>

* **test_descriptions** :  List of text-based descriptions.
* **test_criterias** :  List of test functions. Function input argument names needs to match keys from the data        dictionary.
* **test_title** :  Title of the test.
* **json_output** :  Path into which to write the output results JSON.
* **doc_output** :  Path into which to write the output QMS summary report (docx).

<h3>Raises:</h3>

* **AssertionError** :  If the number of `test_descriptions` and `test_criteria` do not match.



