## AdversarialSample
```python
AdversarialSample(inputs, loss=None, gradients=None, outputs=None, mode=None, epsilon=0.01, clip_low=None, clip_high=None)
```
 This class is to be used to train the model more robust against adversarial attacks (    https://arxiv.org/abs/1412.6572)

#### Args:

* **inputs (str)** :  key of the input to be attacked
* **loss (str)** :  key of the loss value to use in the attack - mutually exclusive with gradients
* **gradients (str)** :  key of the gradients to use in the attack - mutually exclusive with loss
* **outputs (str)** :  key to store the mixed-up input
* **mode (str)** :  what mode to execute in.
* **epsilon (float)** :  epsilon value to perturb the input to create adversarial examples
* **clip_low (float)** :  a minimum value to clip the output by (defaults to min value of data)
* **clip_high (float)** :  a maximum value to clip the output by (defaults to max value of data)    

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