

### get_batch
```python
get_batch(path, batch_size=128, is_train=True, test_lang_list=['Kannada', 'Tengwar', 'Aurek-Besh', 'Sylheti', 'Avesta', 'Glagolitic', 'Manipuri', 'Keble', 'Gurmukhi', 'Oriya'])
```
Data generator for training and validation 

#### Args:

* **path (str)** :  Path to folder containing the images.
* **batch_size (int, optional)** :  batch size, defaults to 128.
* **is_train (bool, optional)** :  whether to generate images for training or validation, defaults to True.
* **test_lang_list (list, optional)** :  List of languages in test dataset, defaults to TEST_LANGUAGES.

#### Returns:

* **(dict)** :  Numpy arrays for the pair of images and label specifying whether the image pair belongs to same or different characters.