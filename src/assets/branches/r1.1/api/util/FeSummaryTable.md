## FeSummaryTable
```python
FeSummaryTable(
	name: str,
	fe_id: fastestimator.util.util.FEID,
	target_type: Type,
	path: Union[NoneType, str, pylatex.base_classes.latex_object.LatexObject]=None,
	kwargs: Union[Dict[str, Any], NoneType]=None, **fields: Any
)
```
A class containing summaries of traceability information.

This class is intentionally not @traceable.


#### Args:

* **name** :  The string to be used as the title line in the summary table.
* **fe_id** :  The id of this table, used for cross-referencing from other tables.
* **target_type** :  The type of the object being summarized.
* **path** :  The import path of the object in question. Might be more complicated when methods/functions are involved.
* **kwargs** :  The keyword arguments used to instantiate the object being summarized.
 **fields :  Any other information about the summarized object / function.

### render_table
```python
render_table(
	self,
	doc: pylatex.document.Document,
	name_override: Union[pylatex.base_classes.latex_object.LatexObject, NoneType]=None,
	toc_ref: Union[str, NoneType]=None,
	extra_rows: Union[List[Tuple[str, Any]], NoneType]=None
)
-> None
```
Write this table into a LaTeX document.


#### Args:

* **doc** :  The LaTeX document to be appended to.
* **name_override** :  An optional replacement for this table's name field.
* **toc_ref** :  A reference to be added to the table of contents.
* **extra_rows** :  Any extra rows to be added to the table before the kwargs.