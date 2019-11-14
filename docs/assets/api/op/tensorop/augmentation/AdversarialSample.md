## AdversarialSample
```python
AdversarialSample(inputs, outputs=None, mode=None, epsilon=0.1)
```
This class is to be used to train the model more robust against adversarial attacks (https://arxiv.org/abs/1412.6572)

#### Args:

* **inputs** :  key of the input to be mixed up
* **outputs** :  key to store the mixed-up input
* **mode** :  what mode to execute in.
* **epsilon** :  epsilon value to perturb the input to create adversarial examples

### forward
```python
forward(self, data, state)
```
 Forward method to perform mixup batch augmentation

#### Args:

* **data** :  Batch data to be augmented
* **state** :  Information about the current execution context.

#### Returns:
            Adversarial example created from perturbing the input data        