## NumpyOp
```python
NumpyOp(inputs:Union[NoneType, str, Iterable[str], Callable]=None, outputs:Union[NoneType, str, Iterable[str]]=None, mode:Union[NoneType, str, Iterable[str]]=None) -> None
```
An Operator class which takes and returns numpy data.    These Operators are used in fe.Pipeline to perform data pre-processing / augmentation.    

### forward
```python
forward(self, data:Union[numpy.ndarray, List[numpy.ndarray]], state:Dict[str, Any]) -> Union[numpy.ndarray, List[numpy.ndarray]]
```
A method which will be invoked in order to transform data.        This method will be invoked on individual elements of data before any batching / axis expansion is performed.

#### Args:

* **data** :  The arrays from the data dictionary corresponding to whatever keys this Op declares as its `inputs`.
* **state** :  Information about the current execution context, for example {"mode" "train"}.

#### Returns:
            The `data` after applying whatever transform this Op is responsible for. It will be written into the data            dictionary based on whatever keys this Op declares as its `outputs`.        