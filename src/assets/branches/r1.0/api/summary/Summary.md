## Summary<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/summary/summary.py/#L19-L45>View source on Github</a>
```python
Summary(
	name: Union[str, NoneType]
)
-> None
```
A summary object that records training history.


<h3>Args:</h3>

* **name** :  Name of the experiment. If None then experiment results will be ignored.

### merge<span class="tag">method of Summary</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/summary/summary.py/#L29-L37>View source on Github</a>
```python
merge(
	self,
	other: 'Summary'
)
```
Merge another `Summary` into this one.


<h4>Args:</h4>

* **other** :  Other `summary` object to be merged.



