## Scheduler<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/schedule/schedule.py/#L24-L44>View source on Github</a>
```python
Scheduler(
	*args, **kwds
)
```
A class which can wrap things like Datasets and Ops to make their behavior epoch-dependent.
    

### get_all_values<span class="tag">method of Scheduler</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/schedule/schedule.py/#L38-L44>View source on Github</a>
```python
get_all_values(
	self
)
-> List[Union[~T, NoneType]]
```
Get a list of all the possible values stored in the `Scheduler`.


<h4>Returns:</h4>
    A list of all the values stored in the `Scheduler`. This may contain None values.

### get_current_value<span class="tag">method of Scheduler</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/schedule/schedule.py/#L27-L36>View source on Github</a>
```python
get_current_value(
	self,
	epoch: int
)
-> Union[~T, NoneType]
```
Fetch whichever of the `Scheduler`s elements is appropriate based on the current epoch.


<h4>Args:</h4>

* **epoch** :  The current epoch.

<h4>Returns:</h4>
    The element from the Scheduler to be used at the given `epoch`. This value might be None.



