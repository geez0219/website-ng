## CSVDataset
```python
CSVDataset(file_path:str, delimiter:str=',', **kwargs) -> None
```
A dataset from a CSV file.    CSVDataset reads entries from a CSV file, where the first row is the header. The root directory of the csv file    may be accessed using dataset.parent_path. This may be useful if the csv contains relative path information    that you want to feed into, say, an ImageReader Op.

#### Args:

* **file_path** :  The (absolute) path to the CSV file.
* **delimiter** :  What delimiter is used by the file.
* **kwargs** :  Other arguments to be passed through to pandas csv reader function. See the pandas docs for details
* **https** : //pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html.    