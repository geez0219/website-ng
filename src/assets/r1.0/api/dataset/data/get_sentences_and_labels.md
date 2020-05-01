

### get_sentences_and_labels
```python
get_sentences_and_labels(path:str) -> Tuple[List[str], List[str], Set[str], Set[str]]
```
Combines tokens into sentences and create vocab set for train data and labels.For simplicity tokens with 'O' entity are omitted.

#### Args:

* **path** :  Path to the downloaded dataset file

#### Returns:
    (sentences, labels, train_vocab, label_vocab)