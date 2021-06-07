## HrefFEID<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/util/latex_util.py/#L142-L176>View source on Github</a>
```python
HrefFEID(
	fe_id: fastestimator.util.util.FEID,
	name: str,
	link_prefix: str='tbl',
	id_in_name: bool=True,
	bold_name: bool=False
)
```
A class to represent a colored and underlined hyperref based on a given fe_id.

This class is intentionally not @traceable.


<h3>Args:</h3>


* **fe_id**: The id used to link this hyperref.

* **name**: A string suffix to be printed as part of the link text.

* **link_prefix**: The prefix for the hyperlink.

* **id_in_name**: Whether to include the id in front of the name text.

* **bold_name**: Whether to bold the name.

