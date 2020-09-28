

### get_batch_size
```python
get_batch_size(
	data: Dict[str, Any]
)
-> int
```
Infer batch size from a batch dictionary. It will ignore all dictionary value with data type that
doesn't have "shape" attribute.


#### Args:

* **data** :  The batch dictionary.

#### Returns:
    batch size.