## UnHadamard
```python
UnHadamard(*args, **kwargs)
```
Convert hadamard encoded class representations into onehot probabilities.


#### Args:

* **inputs** :  Key of the input tensor(s) to be converted.
* **outputs** :  Key of the output tensor(s) as class probabilities.
* **n_classes** :  How many classes are there in the inputs.
* **code_length** :  What code length to use. Will default to the smallest power of 2 which is >= the number of classes.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".