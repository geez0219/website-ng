

### add_summary
```python
add_summary(data_dir, train_prefix, feature_name, feature_dtype, feature_shape, eval_prefix=None, num_train_examples=None, num_eval_examples=None, compression=None)
```
Add summary.json file to existing directory that has TFRecord files.

#### Args:

* **data_dir (str)** :  Folder path where tfrecords are stored.
* **train_prefix (str)** :  The prefix of all training tfrecord files.
* **feature_name (list)** :  Feature name in the tfrecord in a list.
* **feature_dtype (list)** :  Original data type for specific feature, this is used for decoding purpose.
* **feature_shape (list)** :  Original data shape for specific feature, this is used for reshaping purpose.
* **eval_prefix (str, optional)** :  The prefix of all evaluation tfrecord files. Defaults to None.
* **num_train_examples (int, optional)** :  The total number of training examples, if None, it will calculate        automatically. Defaults to None.
* **num_eval_examples (int, optional)** :  The total number of validation examples, if None, it will calculate        automatically. Defaults to None.
* **compression (str, optional)** :  None, 'GZIP' or 'ZLIB'. Defaults to None.