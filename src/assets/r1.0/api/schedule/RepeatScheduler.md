## RepeatScheduler
```python
RepeatScheduler(*args, **kwds)
```
A scheduler which repeats a collection of entries one after another every epoch.    One case where this class would be useful is if you want to perform one version of an Op on even epochs, and a    different version on odd epochs. None values can be used to achieve an end result of skipping an Op every so often.    ```python    s = fe.schedule.RepeatScheduler(["a", "b", "c"])    s.get_current_value(epoch=1)  # "a"    s.get_current_value(epoch=2)  # "b"    s.get_current_value(epoch=3)  # "c"    s.get_current_value(epoch=4)  # "a"    s.get_current_value(epoch=5)  # "b"    ```

#### Args:

* **repeat_list** :  What elements to cycle between every epoch. Note that epochs start counting from 1. To have nothing        happen for a particular epoch, None values may be used.

#### Raises:

* **AssertionError** :  If `repeat_list` is not a List.    