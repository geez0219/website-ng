## Probability
```python
Probability(tensor_op, prob=0.5)
```
Operate a TensorOp with certain probability

#### Args:

* **tensor_op** :  TensorOp instance
* **prob** :  float number which indicates the probability of execution    

### forward
```python
forward(self, data, state)
```
Execute the operator with probability

#### Args:

* **data** :  Tensor to be resized.
* **state** :  Information about the current execution context.

#### Returns:
            output tensor.        