## EpochScheduler<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/schedule/schedule.py/#L81-L141>View source on Github</a>
```python
EpochScheduler(
	epoch_dict: Dict[int, ~T]
)
-> None
```
A scheduler which selects entries based on a specified epoch mapping.

This can be useful for making networks grow over time, or to use more challenging data augmentation as training
progresses.

```python
s = fe.schedule.EpochScheduler({1:"a", 3:"b", 4:None, 100: "c"})
s.get_current_value(epoch=1)  # "a"
s.get_current_value(epoch=2)  # "a"
s.get_current_value(epoch=3)  # "b"
s.get_current_value(epoch=4)  # None
s.get_current_value(epoch=99)  # None
s.get_current_value(epoch=100)  # "c"
```


<h3>Args:</h3>


* **epoch_dict**: A mapping from epoch -> element. For epochs in between keys in the dictionary, the closest prior key will be used to determine which element to return. None values may be used to cause nothing to happen for a particular epoch. 

<h3>Raises:</h3>


* **AssertionError**: If the `epoch_dict` is of the wrong type, or contains invalid keys.

