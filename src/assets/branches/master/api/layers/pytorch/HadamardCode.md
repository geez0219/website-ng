## HadamardCode
```python
HadamardCode(
	in_features: Union[int, List[int]],
	n_classes: int,
	code_length: Union[int, NoneType]=None,
	max_prob: float=0.95,
	power: float=1.0
)
-> None
```
A layer for applying an error correcting code to your outputs.

This class is intentionally not @traceable (models and layers are handled by a different process).

See 'https://papers.nips.cc/paper/9070-error-correcting-output-codes-improve-probability-estimation-and-adversarial-
robustness-of-deep-neural-networks'. Note that for best effectiveness, the model leading into this layer should be
split into multiple independent chunks, whose outputs this layer can combine together in order to perform the code
lookup.

```python
# Use as a drop-in replacement for your softmax layer:
def __init__(self, classes):
    self.fc1 = nn.Linear(1024, 64)
    self.fc2 = nn.Linear(64, classes)
def forward(self, x):
    x = fn.relu(self.fc1(x))
    x = fn.softmax(self.fc2(x), dim=-1)
#   ----- vs ------
def __init__(self, classes):
    self.fc1 = nn.Linear(1024, 64)
    self.fc2 = HadamardCode(64, classes)
def forward(self, x):
    x = fn.relu(self.fc1(x))
    x = self.fc2(x)
```

```python
# Use to combine multiple feature heads for a final output (biggest adversarial hardening benefit):
def __init__(self, classes):
    self.fc1 = nn.ModuleList([nn.Linear(1024, 16) for _ in range(4)])
    self.fc2 = HadamardCode([16]*4, classes)
def forward(self, x):
    x = [fn.relu(fc(x)) for fc in self.fc1]
    x = self.fc2(x)
```


#### Args:

* **in_features** :  How many input features there are (inputs should be of shape (Batch, N) or [(Batch, N), ...]).
* **n_classes** :  How many output classes to map onto.
* **code_length** :  How long of an error correcting code to use. Should be a positive multiple of 2. If not provided,        the smallest power of 2 which is >= `n_outputs` will be used, or 16 if the latter is larger.
* **max_prob** :  The maximum probability that can be assigned to a class. For numeric stability this must be less than        1.0. Intuitively it makes sense to keep this close to 1, but to get adversarial training benefits it should        be noticeably less than 1, for example 0.95 or even 0.8.
* **power** :  The power parameter to be used by Inverse Distance Weighting when transforming Hadamard class distances        into a class probability distribution. A value of 1.0 gives an intuitive mapping to probabilities, but small        values such as 0.25 appear to give slightly better adversarial benefits. Large values like 2 or 3 give        slightly faster convergence at the expense of adversarial performance. Must be greater than zero.

#### Raises:

* **ValueError** :  If `code_length` is invalid.