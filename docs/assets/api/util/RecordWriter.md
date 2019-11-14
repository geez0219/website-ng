## RecordWriter
```python
RecordWriter(train_data, save_dir, validation_data=None, ops=None, write_feature=None, expand_dims=False, max_record_size_mb=300, compression=None)
```
Write data into TFRecords.This class can handle unpaired features. For example, in cycle-gan the hourse and zebra images are unpaired, whichmeans during training you do not have one-to-one correspondance between hourse image and zebra image. When the`RecordWriter` instance is sent to `Pipeline` create random pairs between hourse and zebra images. See the cycle-ganexample in apphub directory.

#### Args:

* **train_data (Union[dict, str])** :  A `dict` that contains train data or a CSV file path. For the CSV file, the        column header will be used as feature name. Under each column in the CSV file the paths to train data should        be provided.
* **save_dir (str)** :  The directory to save the TFRecords.
* **validation_data (Union[dict, str, float], optional)** :  A `dict` that contains validation data, a CSV file path, or        a `float` that is between 0 and 1. For the CSV file, the column header will be used as feature name. Under        each column in the CSV file the paths to validation data should be provided. When this argument is a        `float`, `RecordWriter` will reserve `validation_data` fraction of the `train_data` as validation data.        Defaults to None.
* **ops (obj, optional)** :  Transformation operations before TFRecords creation. Defaults to None.
* **write_feature (str, optional)** :  Users can specify what features they want to write to TFRecords. Defaults to        None.
* **expand_dims (bool, optional)** :  When set to `True`, the first dimension of each feature will be used as batch        dimension. `RecordWriter` will split the batch into single examples and write one example at a time into        TFRecord. Defaults to False.
* **max_record_size_mb (int, optional)** :  Maximum size of single TFRecord file. Defaults to 300 MB.
* **compression (str, optional)** :  Compression type can be `"GZIP"`, `"ZLIB"`, or `""` (no compression). Defaults to        None.

### write
```python
write(self, save_dir=None)
```
Write TFRecods in parallel. Number of processes is set to number of CPU cores.