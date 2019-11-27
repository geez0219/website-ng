

### load_eval_data
```python
load_eval_data(path, lang_list=['Kannada', 'Tengwar', 'Aurek-Besh', 'Sylheti', 'Avesta', 'Glagolitic', 'Manipuri', 'Keble', 'Gurmukhi', 'Oriya'], is_test=False)
```
Load images for evaluation.

#### Args:

* **path (str)** :  Path to evaluation folder.
* **lang_list (list, optional)** :  List of languages in test dataset, defaults to TEST_LANGUAGES
* **is_test (bool, optional)** :  whether to generate images for test dataset, defaults to False.

#### Returns:

* **img_list (list)** :  List of images belonging to each language.