## QMSTest
```python
QMSTest(test_descriptions:Union[str, List[str]], test_criterias:Union[List[Callable], Callable], test_title:str='QMSTest', json_output:str='', doc_output:str='') -> None
```
Automate QMS testing and report generation.

#### Args:

* **test_descriptions** :  List of text-based descriptions.
* **test_criterias** :  List of test functions. Function input argument names needs to match keys from the data            dictionary.
* **test_title** :  Title of the test.
* **json_output** :  Path into which to write the output results JSON.
* **doc_output** :  Path into which to write the output QMS summary report (docx).

#### Raises:

* **AssertionError** :  If the number of `test_descriptions` and `test_criteria` do not match.    