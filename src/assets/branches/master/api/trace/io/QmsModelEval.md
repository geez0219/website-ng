## QmsModelEval
```python
QmsModelEval(test_cases:Union[fastestimator.trace.io.model_eval.TestCase, List[fastestimator.trace.io.model_eval.TestCase]], save_path:str, test_title:Union[str, NoneType]=None, data_id:str=None, intro_page:int=1, abbrev_count:int=10, abbrev_height:float=3.0, version_count:int=5, version_height:float=3.0)
```
GE internal version of ModelEval trace for generation of QMS test report.

The test report of this version will have 3 extra sections: Introduction, Abbreviation, Revision History.
Introduction section provides a fillable text paragraph for documentation of the purpose or introduction of the
model. Abbreviation section provides a table with fillable cells for documentation of term definitions. Revision
History section provides a table with fillable cells for documentation of the revision history.


#### Args:

* **test_cases** :  TestCase object or list of TestCase objects.
* **save_path** :  Where to save the output directory.
* **test_title** :  Title of the test.
* **data_id** :  Data instance ID key. If provided, then per-instance test will return failure instance ID.
* **abbrev_count** :  Number of rows in the table of Abbreviation section.
* **abbrev_height** :  Height of rows in the table of Abbreviation table.
* **version_count** :  Number of rows in the table of Revision History section.
* **version_height** :  Height of rows in the table of Revision History section.

### to_textfield
```python
to_textfield(self, width, height)
```
Generate a fillable text box latex representation.


#### Args:

* **width** :  The width of the text box.
* **height** :  The height of the text box.
* **Retrun** :     The text box latex representation.