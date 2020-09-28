## TestCase
```python
TestCase(
	description: str,
	criteria: Callable[..., Union[bool, numpy.ndarray]],
	aggregate: bool=True,
	fail_threshold: int=0
)
-> None
```
This class defines the test case that the TestReport trace will take to perform auto-testing.


#### Args:

* **description** :  A test description.
* **criteria** :  A function to perform the test. For an aggregate test, `criteria` needs to return True when the test        passes and False when it fails. For a per-instance test, `criteria` needs to return a boolean np.ndarray,        where entries show corresponding test results (True if the test of that data instance passes; False if it        fails).
* **aggregate** :  If True, this test is aggregate type and its `criteria` function will be examined at epoch_end. If        False, this test is per-instance type and its `criteria` function will be examined at batch_end.
* **fail_threshold** :  Threshold of failure instance number to judge the per-instance test as failed or passed. If        the failure number is above this value, then the test fails; otherwise it passes. It can only be set when        `aggregate` is equal to False.

#### Raises:

* **ValueError** :  If user set `fail_threshold` for an aggregate test.

### init_result
```python
init_result(
	self
)
-> None
```
Reset the test result.
        