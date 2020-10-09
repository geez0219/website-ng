## Scheduler<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/schedule/schedule.py/#L22-L42>View source on Github</a>
```python
Scheduler(
	*args, **kwargs
)
```
A class which can wrap things like Datasets and Ops to make their behavior epoch-dependent.
    

---

### get_all_values<span class="tag">method of Scheduler</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/schedule/schedule.py/#L36-L42>View source on Github</a>
```python
get_all_values(
	self
)
-> List[Union[~T, NoneType]]
```
Get a list of all the possible values stored in the `Scheduler`.


<h4>Returns:</h4>

<ul class="return-block"><li>    A list of all the values stored in the <code>Scheduler</code>. This may contain None values.</li></ul>

---

### get_current_value<span class="tag">method of Scheduler</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/schedule/schedule.py/#L25-L34>View source on Github</a>
```python
get_current_value(
	self,
	epoch: int
)
-> Union[~T, NoneType]
```
Fetch whichever of the `Scheduler`s elements is appropriate based on the current epoch.


<h4>Args:</h4>


* **epoch**: The current epoch. 

<h4>Returns:</h4>

<ul class="return-block"><li>    The element from the Scheduler to be used at the given <code>epoch</code>. This value might be None.</li></ul>

