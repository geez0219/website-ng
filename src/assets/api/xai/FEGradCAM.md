## FEGradCAM
```python
FEGradCAM()
```
Perform Grad CAM algorithm for a given input
* **Paper** :  [Grad-CAM Visual Explanations from Deep Networks
* **via Gradient-based Localization](https** : //arxiv.org/abs/1610.02391)

### explain
```python
explain(self, model_input, model, layer_name, class_index, colormap=14)
```
        Compute GradCAM for a specific class index.

#### Args:

* **model_input (tf.tensor)** :  Data to perform the evaluation on.
* **model (tf.keras.Model)** :  tf.keras model to inspect
* **layer_name (str)** :  Targeted layer for GradCAM
* **class_index (int, None)** :  Index of targeted class
* **colormap (int)** :  Used in parent method signature, but ignored here

#### Returns:

* **tf.cams** :  The gradcams        