## Summary
```python
Summary(
	name: Union[str, NoneType],
	system_config: Union[List[fastestimator.util.traceability_util.FeSummaryTable], NoneType]=None
)
-> None
```
A summary object that records training history.

This class is intentionally not @traceable.


#### Args:

* **name** :  Name of the experiment. If None then experiment results will be ignored.
* **system_config** :  A description of the initialization parameters defining the estimator associated with this        experiment.

### merge
```python
merge(
	self,
	other: 'Summary'
)
```
Merge another `Summary` into this one.


#### Args:

* **other** :  Other `summary` object to be merged.