

### get_type
```python
get_type(obj:Any) -> str
```
A function to try and infer the types of data within containers.

```python
x = fe.util.get_type(np.ones((10, 10), dtype='int32'))  # "int32"
x = fe.util.get_type(tf.ones((10, 10), dtype='float16'))  # "<dtype: 'float16'>"
x = fe.util.get_type(torch.ones((10, 10)).type(torch.float))  # "torch.float32"
x = fe.util.get_type([np.ones((10,10)) for i in range(4)])  # "List[float64]"
x = fe.util.get_type(27)  # "int"
```

For container to look into its element's type, its type needs to be either list or tuple, and the return string will
be List[...]. All container elements need to have the same data type becuase it will only check its first element.

```python
x = fe.util.get_type({"a":1, "b":2})  # "dict"
x = fe.util.get_type([1, "a"]) # "List[int]"
x = fe.util.get_type([[[1]]]) # "List[List[List[int]]]"
```


#### Args:

* **obj** :  Data which may be wrapped in some kind of container.

#### Returns:
    A string representation of the data type of the `obj`.