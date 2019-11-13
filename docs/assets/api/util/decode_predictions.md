

### decode_predictions
```python
decode_predictions(predictions, top=3, dictionary=None)
```


#### Args:

* **predictions** :  A batched numpy array of class prediction scores (Batch X Predictions)
* **top** :  How many of the highest predictions to capture
* **dictionary** :  {"<class_idx>" -> "<class_name>"}

#### Returns:
    A right-justified newline-separated array of the top classes and their associated probabilities.    There is one entry in the results array per batch in the input