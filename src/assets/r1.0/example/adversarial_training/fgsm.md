# Adversarial Training Using the Fast Gradient Sign Method (FGSM)

In this example we will demonstrate how to train a model to resist adversarial attacks constructed using the Fast Gradient Sign Method. For more background on adversarial attacks, visit: https://arxiv.org/abs/1412.6572

## Import the required libraries


```python
import tempfile
import os

import numpy as np

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.backend import to_tensor, argmax, to_number
from fastestimator.dataset.data import cifar10
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip
from fastestimator.op.numpyop.univariate import CoarseDropout, Normalize, Onehot
from fastestimator.op.tensorop import Average
from fastestimator.op.tensorop.gradient import Watch, FGSM
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy
from fastestimator.util import ImgData
```


```python
# training parameters
epsilon=0.04  # The strength of the adversarial attack
epochs=10
batch_size=50
max_steps_per_epoch=None
save_dir=tempfile.mkdtemp()
```

## Step 1 - Data and `Pipeline` preparation
In this step, we will load Cifar10 training and validation datasets and prepare FastEstimator's pipeline.

### Load dataset 
We use fastestimator API to load the Cifar10 dataset and get the test set by splitting 50% evaluation set. 


```python
from fastestimator.dataset.data import cifar10

train_data, eval_data = cifar10.load_data()
test_data = eval_data.split(0.5)
```

### Prepare the `Pipeline`
We will use a simple pipeline that just normalizes the images


```python
pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
        ])
```

## Step 2 - `Network` construction

### Model Construction
Here we will leverage the LeNet implementation built in to FastEstimator


```python
model = fe.build(model_fn=lambda: LeNet(input_shape=(32, 32, 3)), optimizer_fn="adam", model_names="adv_model")
```

### `Network` defintion
This is where the adversarial attack will be implemented. To perform an FGSM attack, we first need to monitor gradients with respect to the input image. This can be accomplished in FastEstimator using the `Watch` TensorOp. We then will run the model forward once, compute the loss, and then pass the loss value into the `FGSM` tensorOp in order to create an adversarial image. We will then run the adversarial image through the model, compute the loss again, and average the two results together in order to update the model. 


```python
network = fe.Network(ops=[
        Watch(inputs="x"),
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="base_ce"),
        FGSM(data="x", loss="base_ce", outputs="x_adverse", epsilon=epsilon),
        ModelOp(model=model, inputs="x_adverse", outputs="y_pred_adv"),
        CrossEntropy(inputs=("y_pred_adv", "y"), outputs="adv_ce"),
        Average(inputs=("base_ce", "adv_ce"), outputs="avg_ce"),
        UpdateOp(model=model, loss_name="avg_ce")
    ])
```

## Step 3 - `Estimator` definition and training
In this step, we define the `Estimator` to connect the `Network` with `Pipeline` and set the `traces` which will compute accuracy (Accuracy), and save the best model (BestModelSaver) along the way. We will compute accuracy both with respect to the clean input images ('clean accuracy') as well as with respect to the adversarial input images ('adversarial accuracy'). At the end, we use `Estimator.fit` to trigger the training.


```python
traces = [
    Accuracy(true_key="y", pred_key="y_pred", output_name="clean_accuracy"),
    Accuracy(true_key="y", pred_key="y_pred_adv", output_name="adversarial_accuracy"),
    BestModelSaver(model=model, save_dir=save_dir, metric="base_ce", save_best_mode="min"),
]
estimator = fe.Estimator(pipeline=pipeline,
                         network=network,
                         epochs=epochs,
                         traces=traces,
                         max_steps_per_epoch=max_steps_per_epoch,
                         monitor_names=["base_ce", "adv_ce"],
                         log_steps=1000)
```


```python
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; adv_model_lr: 0.001; 
    FastEstimator-Train: step: 1; adv_ce: 2.4914558; base_ce: 2.3271847; avg_ce: 2.4093204; 
    FastEstimator-Train: step: 1000; adv_ce: 1.6937535; base_ce: 1.29402; avg_ce: 1.4938867; steps/sec: 21.3; 
    FastEstimator-Train: step: 1000; epoch: 1; epoch_time: 50.27 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpddgcxg9j/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 1000; epoch: 1; avg_ce: 1.4610062; base_ce: 1.2934717; adv_ce: 1.6285404; min_avg_ce: 1.4610062; since_best: 0; clean_accuracy: 0.534; adversarial_accuracy: 0.3746; 
    FastEstimator-Train: step: 2000; adv_ce: 1.5694242; base_ce: 1.1551845; avg_ce: 1.3623043; steps/sec: 31.16; 
    FastEstimator-Train: step: 2000; epoch: 2; epoch_time: 32.09 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpddgcxg9j/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 2000; epoch: 2; avg_ce: 1.336678; base_ce: 1.138487; adv_ce: 1.5348687; min_avg_ce: 1.336678; since_best: 0; clean_accuracy: 0.6036; adversarial_accuracy: 0.4276; 
    FastEstimator-Train: step: 3000; adv_ce: 1.6407557; base_ce: 1.1609461; avg_ce: 1.4008509; steps/sec: 32.0; 
    FastEstimator-Train: step: 3000; epoch: 3; epoch_time: 31.25 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpddgcxg9j/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 3000; epoch: 3; avg_ce: 1.3007246; base_ce: 1.0795952; adv_ce: 1.5218539; min_avg_ce: 1.3007246; since_best: 0; clean_accuracy: 0.6182; adversarial_accuracy: 0.4392; 
    FastEstimator-Train: step: 4000; adv_ce: 1.6417248; base_ce: 1.1229622; avg_ce: 1.3823435; steps/sec: 33.29; 
    FastEstimator-Train: step: 4000; epoch: 4; epoch_time: 30.04 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpddgcxg9j/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 4000; epoch: 4; avg_ce: 1.2547151; base_ce: 1.0172701; adv_ce: 1.4921603; min_avg_ce: 1.2547151; since_best: 0; clean_accuracy: 0.6396; adversarial_accuracy: 0.4416; 
    FastEstimator-Train: step: 5000; adv_ce: 1.4567461; base_ce: 0.9144764; avg_ce: 1.1856112; steps/sec: 30.18; 
    FastEstimator-Train: step: 5000; epoch: 5; epoch_time: 33.14 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpddgcxg9j/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 5000; epoch: 5; avg_ce: 1.2455269; base_ce: 0.9953368; adv_ce: 1.495717; min_avg_ce: 1.2455269; since_best: 0; clean_accuracy: 0.6548; adversarial_accuracy: 0.4488; 
    FastEstimator-Train: step: 6000; adv_ce: 1.194348; base_ce: 0.7395498; avg_ce: 0.96694887; steps/sec: 30.23; 
    FastEstimator-Train: step: 6000; epoch: 6; epoch_time: 33.08 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpddgcxg9j/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 6000; epoch: 6; avg_ce: 1.2079716; base_ce: 0.9540632; adv_ce: 1.4618802; min_avg_ce: 1.2079716; since_best: 0; clean_accuracy: 0.6722; adversarial_accuracy: 0.4594; 
    FastEstimator-Train: step: 7000; adv_ce: 1.5178115; base_ce: 1.0440608; avg_ce: 1.2809362; steps/sec: 28.51; 
    FastEstimator-Train: step: 7000; epoch: 7; epoch_time: 35.07 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpddgcxg9j/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 7000; epoch: 7; avg_ce: 1.1914748; base_ce: 0.936576; adv_ce: 1.4463739; min_avg_ce: 1.1914748; since_best: 0; clean_accuracy: 0.6804; adversarial_accuracy: 0.4718; 
    FastEstimator-Train: step: 8000; adv_ce: 1.4089589; base_ce: 0.87968165; avg_ce: 1.1443202; steps/sec: 28.66; 
    FastEstimator-Train: step: 8000; epoch: 8; epoch_time: 34.9 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpddgcxg9j/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 8000; epoch: 8; avg_ce: 1.1998512; base_ce: 0.9261531; adv_ce: 1.473549; min_avg_ce: 1.1914748; since_best: 1; clean_accuracy: 0.6838; adversarial_accuracy: 0.4634; 
    FastEstimator-Train: step: 9000; adv_ce: 1.412882; base_ce: 0.9263946; avg_ce: 1.1696383; steps/sec: 29.12; 
    FastEstimator-Train: step: 9000; epoch: 9; epoch_time: 34.35 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpddgcxg9j/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 9000; epoch: 9; avg_ce: 1.17942; base_ce: 0.90814054; adv_ce: 1.4506993; min_avg_ce: 1.17942; since_best: 0; clean_accuracy: 0.6832; adversarial_accuracy: 0.4738; 
    FastEstimator-Train: step: 10000; adv_ce: 1.5417409; base_ce: 0.9920763; avg_ce: 1.2669086; steps/sec: 29.69; 
    FastEstimator-Train: step: 10000; epoch: 10; epoch_time: 33.68 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpddgcxg9j/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 10000; epoch: 10; avg_ce: 1.1849567; base_ce: 0.90033144; adv_ce: 1.4695816; min_avg_ce: 1.17942; since_best: 1; clean_accuracy: 0.6868; adversarial_accuracy: 0.4618; 
    FastEstimator-Finish: step: 10000; total_time: 365.98 sec; adv_model_lr: 0.001; 


## Model Testing
Let's start by re-loading the weights from the best model, since the model may have overfit during training



```python
model.load_weights(os.path.join(save_dir, "adv_model_best_base_ce.h5"))
```


```python
estimator.test()
```

    FastEstimator-Test: epoch: 10; clean_accuracy: 0.6932; adversarial_accuracy: 0.4674; 


In spite of our training the network using adversarially crafted images, the adversarial attack is still effective at reducing the accuracy of the network. This does not, however, mean that the efforts were wasted. 

# Comparison vs Network without Adversarial Training
To see whether training using adversarial hardening was actually useful, we will compare it to a network which is trained without considering any adversarial images. The setup will be similar to before, but we will only use the adversarial images for evaluation purposes and so the second `CrossEntropy` Op as well as the `Average` Op can be omitted.  


```python
clean_model = fe.build(model_fn=lambda: LeNet(input_shape=(32, 32, 3)), optimizer_fn="adam", model_names="clean_model")
clean_network = fe.Network(ops=[
        Watch(inputs="x"),
        ModelOp(model=clean_model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="base_ce"),
        FGSM(data="x", loss="base_ce", outputs="x_adverse", epsilon=epsilon, mode="!train"),
        ModelOp(model=clean_model, inputs="x_adverse", outputs="y_pred_adv", mode="!train"),
        UpdateOp(model=clean_model, loss_name="base_ce")
    ])
clean_traces = [
    Accuracy(true_key="y", pred_key="y_pred", output_name="clean_accuracy"),
    Accuracy(true_key="y", pred_key="y_pred_adv", output_name="adversarial_accuracy"),
    BestModelSaver(model=clean_model, save_dir=save_dir, metric="base_ce", save_best_mode="min"),
]
clean_estimator = fe.Estimator(pipeline=pipeline,
                         network=clean_network,
                         epochs=epochs,
                         traces=clean_traces,
                         max_steps_per_epoch=max_steps_per_epoch,
                         log_steps=1000)
clean_estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; clean_model_lr: 0.001; 
    FastEstimator-Train: step: 1; base_ce: 2.3439963; 
    FastEstimator-Train: step: 1000; base_ce: 1.3183937; steps/sec: 76.8; 
    FastEstimator-Train: step: 1000; epoch: 1; epoch_time: 13.41 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpddgcxg9j/clean_model_best_base_ce.h5
    FastEstimator-Eval: step: 1000; epoch: 1; base_ce: 1.152839; min_base_ce: 1.152839; since_best: 0; clean_accuracy: 0.5848; adversarial_accuracy: 0.283; 
    FastEstimator-Train: step: 2000; base_ce: 0.8904332; steps/sec: 74.57; 
    FastEstimator-Train: step: 2000; epoch: 2; epoch_time: 13.41 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpddgcxg9j/clean_model_best_base_ce.h5
    FastEstimator-Eval: step: 2000; epoch: 2; base_ce: 0.94625115; min_base_ce: 0.94625115; since_best: 0; clean_accuracy: 0.6774; adversarial_accuracy: 0.2842; 
    FastEstimator-Train: step: 3000; base_ce: 1.0784993; steps/sec: 71.43; 
    FastEstimator-Train: step: 3000; epoch: 3; epoch_time: 14.01 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpddgcxg9j/clean_model_best_base_ce.h5
    FastEstimator-Eval: step: 3000; epoch: 3; base_ce: 0.8710386; min_base_ce: 0.8710386; since_best: 0; clean_accuracy: 0.7002; adversarial_accuracy: 0.2942; 
    FastEstimator-Train: step: 4000; base_ce: 0.765074; steps/sec: 68.29; 
    FastEstimator-Train: step: 4000; epoch: 4; epoch_time: 14.64 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpddgcxg9j/clean_model_best_base_ce.h5
    FastEstimator-Eval: step: 4000; epoch: 4; base_ce: 0.84777266; min_base_ce: 0.84777266; since_best: 0; clean_accuracy: 0.711; adversarial_accuracy: 0.2786; 
    FastEstimator-Train: step: 5000; base_ce: 0.60268235; steps/sec: 69.28; 
    FastEstimator-Train: step: 5000; epoch: 5; epoch_time: 14.43 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpddgcxg9j/clean_model_best_base_ce.h5
    FastEstimator-Eval: step: 5000; epoch: 5; base_ce: 0.79990387; min_base_ce: 0.79990387; since_best: 0; clean_accuracy: 0.7258; adversarial_accuracy: 0.2724; 
    FastEstimator-Train: step: 6000; base_ce: 0.5229054; steps/sec: 74.48; 
    FastEstimator-Train: step: 6000; epoch: 6; epoch_time: 13.43 sec; 
    FastEstimator-Eval: step: 6000; epoch: 6; base_ce: 0.8224955; min_base_ce: 0.79990387; since_best: 1; clean_accuracy: 0.723; adversarial_accuracy: 0.2574; 
    FastEstimator-Train: step: 7000; base_ce: 0.73349226; steps/sec: 74.23; 
    FastEstimator-Train: step: 7000; epoch: 7; epoch_time: 13.47 sec; 
    FastEstimator-ModelSaver: saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpddgcxg9j/clean_model_best_base_ce.h5
    FastEstimator-Eval: step: 7000; epoch: 7; base_ce: 0.78509855; min_base_ce: 0.78509855; since_best: 0; clean_accuracy: 0.7368; adversarial_accuracy: 0.2574; 
    FastEstimator-Train: step: 8000; base_ce: 0.5768247; steps/sec: 74.07; 
    FastEstimator-Train: step: 8000; epoch: 8; epoch_time: 13.5 sec; 
    FastEstimator-Eval: step: 8000; epoch: 8; base_ce: 0.82012117; min_base_ce: 0.78509855; since_best: 1; clean_accuracy: 0.7278; adversarial_accuracy: 0.256; 
    FastEstimator-Train: step: 9000; base_ce: 0.4964206; steps/sec: 70.89; 
    FastEstimator-Train: step: 9000; epoch: 9; epoch_time: 14.11 sec; 
    FastEstimator-Eval: step: 9000; epoch: 9; base_ce: 0.81251466; min_base_ce: 0.78509855; since_best: 2; clean_accuracy: 0.7358; adversarial_accuracy: 0.234; 
    FastEstimator-Train: step: 10000; base_ce: 0.61384237; steps/sec: 70.51; 
    FastEstimator-Train: step: 10000; epoch: 10; epoch_time: 14.19 sec; 
    FastEstimator-Eval: step: 10000; epoch: 10; base_ce: 0.88168806; min_base_ce: 0.78509855; since_best: 3; clean_accuracy: 0.7332; adversarial_accuracy: 0.2268; 
    FastEstimator-Finish: step: 10000; total_time: 157.43 sec; clean_model_lr: 0.001; 


As before, we will reload the best weights and the test the model


```python
clean_model.load_weights(os.path.join(save_dir, "clean_model_best_base_ce.h5"))
```


```python
print("Normal Network:")
normal_results = clean_estimator.test("normal")
print("The whitebox FGSM attack reduced accuracy by {:.2f}".format(list(normal_results.history['test']['clean_accuracy'].values())[0] - list(normal_results.history['test']['adversarial_accuracy'].values())[0]))
print("-----------")
print("Adversarially Trained Network:")
adversarial_results = estimator.test("adversarial")
print("The whitebox FGSM attack reduced accuracy by {:.2f}".format(list(adversarial_results.history['test']['clean_accuracy'].values())[0] - list(adversarial_results.history['test']['adversarial_accuracy'].values())[0]))
print("-----------")
```

    Normal Network:
    FastEstimator-Test: epoch: 10; clean_accuracy: 0.7318; adversarial_accuracy: 0.2434; 
    The whitebox FGSM attack reduced accuracy by 0.49
    -----------
    Adversarially Trained Network:
    FastEstimator-Test: epoch: 10; clean_accuracy: 0.6932; adversarial_accuracy: 0.4674; 
    The whitebox FGSM attack reduced accuracy by 0.23
    -----------


As we can see, the normal network is significantly less robust against adversarial attacks than the one which was trained to resist them. The downside is that the adversarial network requires more epochs of training to converge, and the training steps take about twice as long since they require two forward pass operations. It is also interesting to note that as the regular model was training, it actually saw progressively worse adversarial accuracy. This may be an indication that the network is developing very brittle decision boundaries. 

## Visualizing Adversarial Samples
Lets visualize some images generated by these adversarial attacks to make sure that everything is working as we would expect. The first step is to get some sample data from the pipeline:


```python
class_dictionary = {
    0: "airplane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}
batch = pipeline.get_results(mode="test")
```

Now let's run our sample data through the network and then visualize the results


```python
batch = clean_network.transform(batch, mode="test")
```


```python
n_samples = 10
y = np.array([class_dictionary[clazz.item()] for clazz in to_number(batch["y"][0:n_samples])])
y_pred = np.array([class_dictionary[clazz.item()] for clazz in to_number(argmax(batch["y_pred"], axis=1)[0:n_samples])])
y_adv = np.array([class_dictionary[clazz.item()] for clazz in to_number(argmax(batch["y_pred_adv"], axis=1)[0:n_samples])])
img = ImgData(x=batch["x"][0:n_samples], x_adverse=batch["x_adverse"][0:n_samples], y=y, y_pred=y_pred, y_adv=y_adv)
fig = img.paint_figure()
```


![png](assets/example/adversarial_training/fgsm_files/fgsm_29_0.png)


As you can see, the adversarial images appear very similar to the unmodified images, and yet they are often able to modify the class predictions of the network. Note that if a network's prediction is already wrong, the attack is unlikely to change the incorrect prediction, but rather to increase the model's confidence in its incorrect prediction. 
