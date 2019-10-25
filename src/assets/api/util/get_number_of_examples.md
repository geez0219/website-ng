

### get_number_of_examples
```python
get_number_of_examples(file_path, show_warning=True, compression=None)
```
Return number of examples in one TFRecord.

#### Args:

* **file_path (str)** :  Path of TFRecord file.
* **show_warning (bool)** :  Whether to display warning messages.
* **compression (str)** :  TFRecord compression type `None`, `'GZIP'`, or `'ZLIB'`.

#### Returns:
    Number of examples in the TFRecord.