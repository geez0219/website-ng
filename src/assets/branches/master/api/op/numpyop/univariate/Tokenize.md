## Tokenize
```python
Tokenize(*args, **kwargs)
```
Split the sequences into tokens.

Tokenize split the document/sequence into tokens and at the same time perform additional operations on tokens if
defined in the passed function object. By default, tokenize only splits the sequences into tokens.


#### Args:

* **inputs** :  Key(s) of sequences to be tokenized.
* **outputs** :  Key(s) of sequences that are tokenized.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **tokenize_fn** :  Tokenization function object.
* **to_lower_case** :  Whether to convert tokens to lowercase.