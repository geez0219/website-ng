## TestCase
```python
TestCase(*args, **kwargs)
```
This class defines the test case that ModelEval trace will take to perform auto-testing.


#### Args:

* **description** :  A test description.
* **criteria** :  A function to perform the test. For an aggregate test, <criteria> needs to return True when the test        passes and False when it fails. For a per-instance test, <criteria> needs to return an ndarray of bool,        where entries show corresponding test results. (True if the test of that data instance passes; False if it        fails).
* **aggregate** :  If True, this test is aggregate type and its <criteria> function will be examined at epoch_end. If        False, this test is per-instance type and its <criteria> function will be examined at batch_end.
* **fail_threshold** :  Thershold of failure instance number to judge the per-instance test as failed or passed. If        the failure number is above this value, then the test fails; otherwise it passes. It only has effect when        <aggregate> is equal to False.

### init_result
```python
init_result(self) -> None
```
Reset test result.
        