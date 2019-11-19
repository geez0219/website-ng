# CIFAR10 Image Classification Using LeNet With Adversarial Training

In this tutorial, we are going to walk through the logic in `lenet_cifar10_adversarial.py` shown below and provide step-by-step instructions.


```python
!cat lenet_cifar10_adversarial.py
```

## Step 1: Prepare training and evaluation dataset, create FastEstimator `Pipeline`

`Pipeline` can take both data in memory and data in disk. In this example, we are going to use data in memory by loading data with `tf.keras.datasets.cifar10`


```python
import tensorflow as tf

(x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.cifar10.load_data()
print("train image shape is {}".format(x_train.shape))
print("train label shape is {}".format(y_train.shape))
print("eval image shape is {}".format(x_eval.shape))
print("eval label shape is {}".format(y_eval.shape))
```

    train image shape is (50000, 32, 32, 3)
    train label shape is (50000, 1)
    eval image shape is (10000, 32, 32, 3)
    eval label shape is (10000, 1)


For in-memory data in `Pipeline`, the data format should be a nested dictionary like: {"mode1": {"feature1": numpy_array, "feature2": numpy_array, ...}, ...}. Each `mode` can be either `train` or `eval`, in our case, we have both `train` and `eval`.  `feature` is the feature name, in our case, we have `x` and `y`.


```python
data = {"train": {"x": x_train, "y": y_train}, "eval": {"x": x_eval, "y": y_eval}}
```


```python
#Parameters
epochs = 15
batch_size = 50
steps_per_epoch = None
validation_steps = None
num_test_samples = 10000
```

Now we are ready to define `Pipeline`, we want to apply a `Minmax` online preprocessing to the image feature `x` for both training and evaluation:


```python
import fastestimator as fe
from fastestimator.op.tensorop import Minmax

pipeline = fe.Pipeline(batch_size=batch_size, data=data, ops=Minmax(inputs="x", outputs="x"))
```

## Step 2: Prepare model, create FastEstimator `Network`

First, we have to define the network architecture in `tf.keras.Model` or `tf.keras.Sequential`, for a popular architecture like LeNet, FastEstimator has it implemented already in [fastestimator.architecture.lenet](https://github.com/fastestimator/fastestimator/blob/master/fastestimator/architecture/lenet.py).  After defining the architecture, users are expected to feed the architecture definition and its associated model name, optimizer and loss name (default to be 'loss') to `FEModel`.


```python
from fastestimator.architecture import LeNet

model = fe.build(model_def=lambda: LeNet(input_shape=x_train.shape[1:], classes=10), model_name="LeNet", 
                 optimizer="adam", loss_name="loss")
```

We can now define a simple `Network`: given with a batch data with key `x` and `y`, we have to work our way to `loss` with series of operators.  `ModelOp` is an operator that contains a model.


```python
from fastestimator.op.tensorop import ModelOp, SparseCategoricalCrossentropy

simple_network = fe.Network(ops=[ModelOp(inputs="x", model=model, outputs="y_pred"), 
                                 SparseCategoricalCrossentropy(y_pred="y_pred", y_true="y", outputs="loss")])
```

One advantage of `FastEstimator`, though, is that it is easy to construct much more complicated graphs. In this example, we want to conduct training by generating adversarially perturbed images and training against them, since this has been shown to make neural networks more robust against future [attacks](https://arxiv.org/abs/1412.6572). To achieve this in `FastEstimator`, we start by running the input through the model op and computing loss as before, but this time the `ModelOp` has the track_input flag set to `True` in order to indicate that gradients should be computed with respect to the input image in addition to the model weights. The network then generates an adversarial sample image using the `AdversarialSample` augmentation module, and runs that image through the model. Finally, the model is trained using an average of the raw loss and adversarial loss. Note that the adversarial part of the process needs only be done during training (not evaluation) and so the `mode` of these final four operations is set to 'train'.  


```python
from fastestimator.op.tensorop import AdversarialSample, Average

pipeline2 = fe.Pipeline(batch_size=batch_size, data=data, ops=Minmax(inputs="x", outputs="x"))
model2 = fe.build(model_def=lambda: LeNet(input_shape=x_train.shape[1:], classes=10), 
                  model_name="LeNet", optimizer="adam", loss_name="loss")

adversarial_network = fe.Network(ops=[
        ModelOp(inputs="x", model=model2, outputs="y_pred", track_input=True),
        SparseCategoricalCrossentropy(y_true="y", y_pred="y_pred", outputs="loss"),
        AdversarialSample(inputs=("loss", "x"), outputs="x_adverse", epsilon=0.01, mode="train"),
        ModelOp(inputs="x_adverse", model=model2, outputs="y_pred_adverse", mode="train"),
        SparseCategoricalCrossentropy(y_true="y", y_pred="y_pred_adverse", outputs="adverse_loss", mode="train"),
        Average(inputs=("loss", "adverse_loss"), outputs="loss", mode="train")
    ])
```

## Step 3: Configure training, create `Estimator`

During the training loop, we want to: 1) measure accuracy for data data 2) save the model with lowest valdiation loss. The `Trace` class is used for anything related to the training loop, and we will need to import the `Accuracy` and `ModelSaver` traces.


```python
import tempfile
import os
from fastestimator.trace import Accuracy, ModelSaver

base_dir = tempfile.mkdtemp()
simple_save_dir = os.path.join(base_dir, 'simple')
adversarial_save_dir = os.path.join(base_dir, 'adverse')

simple_traces = [Accuracy(true_key="y", pred_key="y_pred", output_name='acc'),
                 ModelSaver(model_name="LeNet", save_dir=simple_save_dir, save_best=True)]

adversarial_traces = [Accuracy(true_key="y", pred_key="y_pred", output_name='acc'),
                      ModelSaver(model_name="LeNet", save_dir=adversarial_save_dir, save_best=True)]
```

Now we can define the `Estimator` and specify the training configuation. We will create estimators for both the simple and adversarial networks in order to compare their performances.


```python
simple_estimator = fe.Estimator(network=simple_network, 
                                pipeline=pipeline, 
                                epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_steps=validation_steps,
                                traces=simple_traces, 
                                log_steps=500)

adversarial_estimator = fe.Estimator(network=adversarial_network, 
                                     pipeline=pipeline2, 
                                     epochs=epochs,
                                     steps_per_epoch=steps_per_epoch,
                                     validation_steps=validation_steps,
                                     traces=adversarial_traces, 
                                     log_steps=500)
```

## Step 4: Training

We'll start by training the regular network (takes about 7.7 minutes on a 2015 MacBookPro CPU - 2.5 GHz Intel Core i7). The network should attain an evaluation accuracy around 71%


```python
simple_estimator.fit()
```

Next we train the network adversarially. This process takes longer (about 17 minutes) since it requires two different gradient computations and model evaluations per forward step rather than one. It is also slower to converge since the training process is more difficult, though should also get to around 71% evaluation accuracy.


```python
adversarial_estimator.fit()
```

## Step 5: Inferencing and Adversarial Attacks

After training, the model is saved to a temporary folder. We can load the model from file and do inferencing on a sample image.


```python
simple_model_path = os.path.join(simple_save_dir, 'LeNet_best_loss.h5')
simple_model = tf.keras.models.load_model(simple_model_path, compile=False)

adversarial_model_path = os.path.join(adversarial_save_dir, 'LeNet_best_loss.h5')
adversarial_model = tf.keras.models.load_model(adversarial_model_path, compile=False)
```

Lets consider a few images from the evaluation dataset and see how the networks respond to adversarial attacks


```python
import matplotlib.pyplot as plt
import numpy as np
from fastestimator.interpretation import show_image
from fastestimator.op.tensorop import Minmax

minmax = Minmax()
num_vis = 10
num_samples = num_test_samples

fig, axis = plt.subplots(1, num_vis, figsize=(21, 3))
sample_images = tf.stack([minmax.forward(tf.constant(x), {}) for x in x_eval[0:num_samples]])
sample_labels = tf.constant(y_eval[0:num_samples])
for idx in range(num_vis):
    show_image(sample_images[idx], axis=axis[idx])

class_dictionary = {
    0: "airplane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

print("True Labels:               [{}]".format(
    ', '.join(['{:<8}' for _ in range(num_vis)])).format(*[class_dictionary[x[0].numpy()] for x in sample_labels][0:num_vis]))
simple_prediction_score = simple_model.predict(sample_images)
simple_accuracy = 1.0 - np.sum(np.not_equal(np.argmax(simple_prediction_score, axis=1), tf.reshape(sample_labels, (num_samples,)))) / num_samples
print("Simple Model Predicts:     [{}] ({:2.1%} accuracy over {} images)".format(
    ', '.join(['{:<8}' for _ in range(num_vis)]), simple_accuracy, num_samples).format(
    *[class_dictionary[x] for x in np.argmax(simple_prediction_score, axis=1)][0:num_vis]))
adversarial_prediction_score = adversarial_model.predict(sample_images)
adversarial_accuracy = 1.0 - np.sum(np.not_equal(np.argmax(adversarial_prediction_score, axis=1), tf.reshape(sample_labels, (num_samples,)))) / num_samples
print("Adversarial Model Predicts:[{}] ({:2.1%} accuracy over {} images)".format(
    ', '.join(['{:<8}' for _ in range(num_vis)]), adversarial_accuracy, num_samples).format(
    *[class_dictionary[x] for x in np.argmax(adversarial_prediction_score, axis=1)][0:num_vis]))
```

    True Labels:               [cat     , ship    , ship    , airplane, frog    , frog    , car     , frog    , cat     , car     ]
    Simple Model Predicts:     [cat     , ship    , ship    , airplane, deer    , frog    , car     , bird    , cat     , car     ] (71.3% accuracy over 10000 images)
    Adversarial Model Predicts:[cat     , ship    , ship    , airplane, deer    , frog    , car     , deer    , cat     , car     ] (71.1% accuracy over 10000 images)



![png](lenet_cifar10_adversarial_files/lenet_cifar10_adversarial_32_1.png)


As we can see, both the simple model and the one trained against adversarial samples correctly identify a majority of the evaluation images, with a population accuracy around 70% each. Now, to create the adversarial versions of the images, we'll simulate the adversarial augmentation object



```python
def attack(images, model, ground_truth, epsilon):
    loss_obj = tf.losses.SparseCategoricalCrossentropy(reduction='none')
    with tf.GradientTape() as tape:
        tape.watch(images)
        pred = model(images, training=False)
        loss = loss_obj(ground_truth, pred)
    gradients = tape.gradient(loss, images)
    adverse_images = tf.clip_by_value(images + epsilon * tf.sign(gradients),
                                      tf.reduce_min(images),
                                      tf.reduce_max(images))
    return adverse_images
```

First we will generate adversarial images by inspecting the gradients of the simple model, and see how well the two models can resist the attack


```python
adverse_images = attack(sample_images, simple_model, sample_labels, 0.01)

fig, axis = plt.subplots(1, num_vis, figsize=(21, 3))
for idx in range(num_vis):
    show_image(adverse_images[idx], axis=axis[idx])
    
print("True Labels:               [{}]".format(
    ', '.join(['{:<8}' for _ in range(num_vis)])).format(*[class_dictionary[x[0].numpy()] for x in sample_labels][0:num_vis]))
simple_prediction_score = simple_model.predict(adverse_images)
simple_accuracy_w = 1.0 - np.sum(np.not_equal(np.argmax(simple_prediction_score, axis=1), tf.reshape(sample_labels, (num_samples,)))) / num_samples
print("Simple Model Predicts:     [{}] ({:2.1%} accuracy over {} images)".format(
    ', '.join(['{:<8}' for _ in range(num_vis)]), simple_accuracy_w, num_samples).format(
    *[class_dictionary[x] for x in np.argmax(simple_prediction_score, axis=1)][0:num_vis]))
adversarial_prediction_score = adversarial_model.predict(adverse_images)
adversarial_accuracy_b = 1.0 - np.sum(np.not_equal(np.argmax(adversarial_prediction_score, axis=1), tf.reshape(sample_labels, (num_samples,)))) / num_samples
print("Adversarial Model Predicts:[{}] ({:2.1%} accuracy over {} images)".format(
    ', '.join(['{:<8}' for _ in range(num_vis)]), adversarial_accuracy_b, num_samples).format(
    *[class_dictionary[x] for x in np.argmax(adversarial_prediction_score, axis=1)][0:num_vis]))
```

    True Labels:               [cat     , ship    , ship    , airplane, frog    , frog    , car     , frog    , cat     , car     ]
    Simple Model Predicts:     [dog     , ship    , airplane, airplane, deer    , cat     , dog     , deer    , bird    , truck   ] (30.7% accuracy over 10000 images)
    Adversarial Model Predicts:[cat     , ship    , ship    , airplane, deer    , frog    , car     , deer    , cat     , car     ] (66.6% accuracy over 10000 images)



![png](lenet_cifar10_adversarial_files/lenet_cifar10_adversarial_36_1.png)


Even though these adversarially attacked images look basically the same as the original images, the accuracy of the traditionally trained model has dropped to 31.9%. The adversarially trained model also sees a reduction in accuracy, but only to 65.2%. It is, however, an incomplete/unfair comparison since the attack is white-box against the simple network but black-box against the adversarially trained network. Let's now generate adversarial images using the adversarially trainined network instead and see how well the models resist the attack


```python
adverse_images = attack(sample_images, adversarial_model, sample_labels, 0.01)

fig, axis = plt.subplots(1, num_vis, figsize=(21, 3))
for idx in range(num_vis):
    show_image(adverse_images[idx], axis=axis[idx])
    
print("True Labels:               [{}]".format(
    ', '.join(['{:<8}' for _ in range(num_vis)])).format(*[class_dictionary[x[0].numpy()] for x in sample_labels][0:num_vis]))
simple_prediction_score = simple_model.predict(adverse_images)
simple_accuracy_b = 1.0 - np.sum(np.not_equal(np.argmax(simple_prediction_score, axis=1), tf.reshape(sample_labels, (num_samples,)))) / num_samples
print("Simple Model Predicts:     [{}] ({:2.1%} accuracy over {} images)".format(
    ', '.join(['{:<8}' for _ in range(num_vis)]), simple_accuracy_b, num_samples).format(
    *[class_dictionary[x] for x in np.argmax(simple_prediction_score, axis=1)][0:num_vis]))
adversarial_prediction_score = adversarial_model.predict(adverse_images)
adversarial_accuracy_w = 1.0 - np.sum(np.not_equal(np.argmax(adversarial_prediction_score, axis=1), tf.reshape(sample_labels, (num_samples,)))) / num_samples
print("Adversarial Model Predicts:[{}] ({:2.1%} accuracy over {} images)".format(
    ', '.join(['{:<8}' for _ in range(num_vis)]), adversarial_accuracy_w, num_samples).format(
    *[class_dictionary[x] for x in np.argmax(adversarial_prediction_score, axis=1)][0:num_vis]))
```

    True Labels:               [cat     , ship    , ship    , airplane, frog    , frog    , car     , frog    , cat     , car     ]
    Simple Model Predicts:     [cat     , ship    , ship    , airplane, deer    , frog    , car     , deer    , cat     , truck   ] (62.0% accuracy over 10000 images)
    Adversarial Model Predicts:[dog     , airplane, airplane, airplane, deer    , cat     , car     , deer    , cat     , truck   ] (49.5% accuracy over 10000 images)



![png](lenet_cifar10_adversarial_files/lenet_cifar10_adversarial_38_1.png)


Under this attack, the accuracy of the traditionally trained model has dropped to 61.6%. The adversarially trained model meanwhile has its performance reduced to 49.1%. While the raw adversarial accuracy here is now lower than the simple model, the performance loss is significantly less than it was for the simple model in the previous attack. To properly compare the models, the white-box and black-box attacks should be compared pairwise against one another:


```python
print("White box attack vs simple network:      {:2.2%} accuracy".format(simple_accuracy_w - simple_accuracy))
print("White box attack vs adversarial network: {:2.2%} accuracy".format(adversarial_accuracy_w - adversarial_accuracy))
print()
print("Black box attack vs simple network:      {:2.2%} accuracy".format(simple_accuracy_b - simple_accuracy))
print("Black box attack vs adversarial network: {:2.2%} accuracy".format(adversarial_accuracy_b - adversarial_accuracy))
```

    White box attack vs simple network:      -40.61% accuracy
    White box attack vs adversarial network: -21.88% accuracy
    
    Black box attack vs simple network:      -9.38% accuracy
    Black box attack vs adversarial network: -4.70% accuracy


Adversarially attacking the simple network using white-box gradient analysis cost nearly 40 percentage points of accuracy. The same attack conducted against the adversarially trained network cost only around 23 percentage points. Likewise, a blackbox attack against the simple network cost 10 percentage points versus 6.5 against the adversarial network. This shows that the adversarial training process makes a network approximately twice as robust against future adversarial attacks. Whether such benefits are worth the increased training time would, of course, depend on the model deployment use-case. 
