## TensorOp
```python
TensorOp(inputs:Union[NoneType, str, Iterable[str], Callable]=None, outputs:Union[NoneType, str, Iterable[str]]=None, mode:Union[NoneType, str, Iterable[str]]=None) -> None
```
An Operator class which takes and returns tensor data.    These Operators are used in fe.Network to perform graph-based operations like neural network training.    

### forward
```python
forward(self, data:Union[~Tensor, List[~Tensor]], state:Dict[str, Any]) -> Union[~Tensor, List[~Tensor]]
```
A method which will be invoked in order to transform data.        This method will be invoked on batches of data.

#### Args:

* **data** :  The batch from the data dictionary corresponding to whatever keys this Op declares as its `inputs`.
* **state** :  Information about the current execution context, for example {"mode" "train"}.

#### Returns:
            The `data` after applying whatever transform this Op is responsible for. It will be written into the data            dictionary based on whatever keys this Op declares as its `outputs`.        