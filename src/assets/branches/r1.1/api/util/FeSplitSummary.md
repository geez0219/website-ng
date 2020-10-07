## FeSplitSummary<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/util/traceability_util.py/#L117-L145>View source on Github</a>
```python
FeSplitSummary()
```
A class to summarize splits performed on an FE Dataset.

This class is intentionally not @traceable.

### add_split<span class="tag">method of FeSplitSummary</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/util/traceability_util.py/#L126-L133>View source on Github</a>
```python
add_split(
	self,
	parent: Union[fastestimator.util.util.FEID, str],
	fraction: str
)
-> None
```
Record another split on this dataset.


<h4>Args:</h4>

* **parent** :  The id of the parent involved in the split (or 'self' if you are the parent).
* **fraction** :  The string representation of the split fraction that was used.

### dumps<span class="tag">method of FeSplitSummary</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/util/traceability_util.py/#L135-L145>View source on Github</a>
```python
dumps(
	self
)
-> str
```
Generate a LaTeX formatted representation of this object.


<h4>Returns:</h4>
    A LaTeX string representation of this object.



