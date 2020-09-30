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
max_train_steps_per_epoch=None
max_eval_steps_per_epoch=None
save_dir=tempfile.mkdtemp()
```

## Step 1 - Data and `Pipeline` preparation
In this step, we will load CIFAR10 training and validation datasets and prepare FastEstimator's pipeline.

### Load dataset 
We use a FastEstimator API to load the CIFAR10 dataset and then get a test set by splitting 50% of the data off of the evaluation set. 


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
model = fe.build(model_fn=lambda: LeNet(input_shape=(32, 32, 3)), optimizer_fn="adam", model_name="adv_model")
```

### `Network` defintion
This is where the adversarial attack will be implemented. To perform an FGSM attack, we first need to monitor gradients with respect to the input image. This can be accomplished in FastEstimator using the `Watch` TensorOp. We then will run the model forward once, compute the loss, and then pass the loss value into the `FGSM` TensorOp in order to create an adversarial image. We will then run the adversarial image through the model, compute the loss again, and average the two results together in order to update the model. 


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
In this step, we define the `Estimator` to connect the `Network` with the `Pipeline` and set the `traces` which will compute accuracy (`Accuracy`) and save the best model (`BestModelSaver`) along the way. We will compute accuracy both with respect to the clean input images ('clean accuracy') as well as with respect to the adversarial input images ('adversarial accuracy'). At the end, we use `Estimator.fit` to trigger the training.


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
                         max_train_steps_per_epoch=max_train_steps_per_epoch,
                         max_eval_steps_per_epoch=max_eval_steps_per_epoch,
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
                                                                            
    
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 1000; 
    FastEstimator-Train: step: 1; avg_ce: 2.3945074; adv_ce: 2.4872663; base_ce: 2.3017485; 
    FastEstimator-Train: step: 1000; avg_ce: 1.3094263; adv_ce: 1.4686574; base_ce: 1.1501954; steps/sec: 25.77; 
    FastEstimator-Train: step: 1000; epoch: 1; epoch_time: 42.87 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpmwpkk98d/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 1000; epoch: 1; avg_ce: 1.4899004; adv_ce: 1.6584656; base_ce: 1.3213345; clean_accuracy: 0.5408; adversarial_accuracy: 0.3734; since_best_base_ce: 0; min_base_ce: 1.3213345; 
    FastEstimator-Train: step: 2000; avg_ce: 1.143224; adv_ce: 1.3376933; base_ce: 0.9487548; steps/sec: 34.22; 
    FastEstimator-Train: step: 2000; epoch: 2; epoch_time: 29.22 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpmwpkk98d/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 2000; epoch: 2; avg_ce: 1.3468871; adv_ce: 1.5618094; base_ce: 1.1319652; clean_accuracy: 0.5928; adversarial_accuracy: 0.412; since_best_base_ce: 0; min_base_ce: 1.1319652; 
    FastEstimator-Train: step: 3000; avg_ce: 1.379614; adv_ce: 1.6073439; base_ce: 1.1518841; steps/sec: 36.24; 
    FastEstimator-Train: step: 3000; epoch: 3; epoch_time: 27.6 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpmwpkk98d/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 3000; epoch: 3; avg_ce: 1.3020732; adv_ce: 1.5298728; base_ce: 1.074274; clean_accuracy: 0.622; adversarial_accuracy: 0.4274; since_best_base_ce: 0; min_base_ce: 1.074274; 
    FastEstimator-Train: step: 4000; avg_ce: 1.2436087; adv_ce: 1.4758196; base_ce: 1.0113977; steps/sec: 33.11; 
    FastEstimator-Train: step: 4000; epoch: 4; epoch_time: 30.2 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpmwpkk98d/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 4000; epoch: 4; avg_ce: 1.2154673; adv_ce: 1.4576334; base_ce: 0.9733012; clean_accuracy: 0.665; adversarial_accuracy: 0.4618; since_best_base_ce: 0; min_base_ce: 0.9733012; 
    FastEstimator-Train: step: 5000; avg_ce: 1.154286; adv_ce: 1.3962423; base_ce: 0.9123298; steps/sec: 32.78; 
    FastEstimator-Train: step: 5000; epoch: 5; epoch_time: 30.51 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpmwpkk98d/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 5000; epoch: 5; avg_ce: 1.2138638; adv_ce: 1.4704447; base_ce: 0.9572834; clean_accuracy: 0.6696; adversarial_accuracy: 0.4552; since_best_base_ce: 0; min_base_ce: 0.9572834; 
    FastEstimator-Train: step: 6000; avg_ce: 1.1946353; adv_ce: 1.4845756; base_ce: 0.904695; steps/sec: 33.34; 
    FastEstimator-Train: step: 6000; epoch: 6; epoch_time: 29.99 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpmwpkk98d/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 6000; epoch: 6; avg_ce: 1.1873512; adv_ce: 1.4471037; base_ce: 0.9275986; clean_accuracy: 0.6784; adversarial_accuracy: 0.4648; since_best_base_ce: 0; min_base_ce: 0.9275986; 
    FastEstimator-Train: step: 7000; avg_ce: 1.3036005; adv_ce: 1.5895638; base_ce: 1.0176373; steps/sec: 32.39; 
    FastEstimator-Train: step: 7000; epoch: 7; epoch_time: 30.87 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpmwpkk98d/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 7000; epoch: 7; avg_ce: 1.1750631; adv_ce: 1.4450079; base_ce: 0.9051186; clean_accuracy: 0.6934; adversarial_accuracy: 0.4764; since_best_base_ce: 0; min_base_ce: 0.9051186; 
    FastEstimator-Train: step: 8000; avg_ce: 1.1723398; adv_ce: 1.4465908; base_ce: 0.89808875; steps/sec: 32.64; 
    FastEstimator-Train: step: 8000; epoch: 8; epoch_time: 30.64 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpmwpkk98d/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 8000; epoch: 8; avg_ce: 1.1422093; adv_ce: 1.4145594; base_ce: 0.869859; clean_accuracy: 0.702; adversarial_accuracy: 0.477; since_best_base_ce: 0; min_base_ce: 0.869859; 
    FastEstimator-Train: step: 9000; avg_ce: 1.0794711; adv_ce: 1.3604188; base_ce: 0.7985235; steps/sec: 32.8; 
    FastEstimator-Train: step: 9000; epoch: 9; epoch_time: 30.49 sec; 
    FastEstimator-Eval: step: 9000; epoch: 9; avg_ce: 1.1585152; adv_ce: 1.4406301; base_ce: 0.8764003; clean_accuracy: 0.7014; adversarial_accuracy: 0.4746; since_best_base_ce: 1; min_base_ce: 0.869859; 
    FastEstimator-Train: step: 10000; avg_ce: 1.2206926; adv_ce: 1.5394913; base_ce: 0.901894; steps/sec: 33.19; 
    FastEstimator-Train: step: 10000; epoch: 10; epoch_time: 30.13 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpmwpkk98d/adv_model_best_base_ce.h5
    FastEstimator-Eval: step: 10000; epoch: 10; avg_ce: 1.1538234; adv_ce: 1.4526846; base_ce: 0.85496193; clean_accuracy: 0.7084; adversarial_accuracy: 0.4792; since_best_base_ce: 0; min_base_ce: 0.85496193; 
    FastEstimator-Finish: step: 10000; total_time: 330.22 sec; adv_model_lr: 0.001; 


## Model Testing
Let's start by re-loading the weights from the best model, since the model may have overfit during training



```python
model.load_weights(os.path.join(save_dir, "adv_model_best_base_ce.h5"))
```


```python
estimator.test()
```

    FastEstimator-Test: step: 10000; epoch: 10; clean_accuracy: 0.6962; adversarial_accuracy: 0.4758; 


In spite of our training the network using adversarially crafted images, the adversarial attack is still effective at reducing the accuracy of the network. This does not, however, mean that the efforts were wasted. 

# Comparison vs Network without Adversarial Training
To see whether training using adversarial hardening was actually useful, we will compare it to a network which is trained without considering any adversarial images. The setup will be similar to before, but we will only use the adversarial images for evaluation purposes and so the second `CrossEntropy` Op as well as the `Average` Op can be omitted.  


```python
clean_model = fe.build(model_fn=lambda: LeNet(input_shape=(32, 32, 3)), optimizer_fn="adam", model_name="clean_model")
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
                         max_train_steps_per_epoch=max_train_steps_per_epoch,
                         max_eval_steps_per_epoch=max_eval_steps_per_epoch,
                         log_steps=1000)
clean_estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; num_device: 0; logging_interval: 1000; 
    FastEstimator-Train: step: 1; base_ce: 2.3599913; 
    FastEstimator-Train: step: 1000; base_ce: 1.2336738; steps/sec: 81.68; 
    FastEstimator-Train: step: 1000; epoch: 1; epoch_time: 12.53 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpmwpkk98d/clean_model_best_base_ce.h5
    FastEstimator-Eval: step: 1000; epoch: 1; base_ce: 1.1847152; clean_accuracy: 0.5684; adversarial_accuracy: 0.2694; since_best_base_ce: 0; min_base_ce: 1.1847152; 
    FastEstimator-Train: step: 2000; base_ce: 0.81964266; steps/sec: 78.27; 
    FastEstimator-Train: step: 2000; epoch: 2; epoch_time: 12.78 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpmwpkk98d/clean_model_best_base_ce.h5
    FastEstimator-Eval: step: 2000; epoch: 2; base_ce: 0.95957977; clean_accuracy: 0.6652; adversarial_accuracy: 0.2778; since_best_base_ce: 0; min_base_ce: 0.95957977; 
    FastEstimator-Train: step: 3000; base_ce: 0.9629886; steps/sec: 78.25; 
    FastEstimator-Train: step: 3000; epoch: 3; epoch_time: 12.78 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpmwpkk98d/clean_model_best_base_ce.h5
    FastEstimator-Eval: step: 3000; epoch: 3; base_ce: 0.8996327; clean_accuracy: 0.6946; adversarial_accuracy: 0.263; since_best_base_ce: 0; min_base_ce: 0.8996327; 
    FastEstimator-Train: step: 4000; base_ce: 0.7768238; steps/sec: 77.71; 
    FastEstimator-Train: step: 4000; epoch: 4; epoch_time: 12.87 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpmwpkk98d/clean_model_best_base_ce.h5
    FastEstimator-Eval: step: 4000; epoch: 4; base_ce: 0.86543256; clean_accuracy: 0.702; adversarial_accuracy: 0.2576; since_best_base_ce: 0; min_base_ce: 0.86543256; 
    FastEstimator-Train: step: 5000; base_ce: 0.7760873; steps/sec: 77.93; 
    FastEstimator-Train: step: 5000; epoch: 5; epoch_time: 12.83 sec; 
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpmwpkk98d/clean_model_best_base_ce.h5
    FastEstimator-Eval: step: 5000; epoch: 5; base_ce: 0.7983937; clean_accuracy: 0.727; adversarial_accuracy: 0.2664; since_best_base_ce: 0; min_base_ce: 0.7983937; 
    FastEstimator-Train: step: 6000; base_ce: 0.56715065; steps/sec: 78.43; 
    FastEstimator-Train: step: 6000; epoch: 6; epoch_time: 12.75 sec; 
    FastEstimator-Eval: step: 6000; epoch: 6; base_ce: 0.79985946; clean_accuracy: 0.7318; adversarial_accuracy: 0.2704; since_best_base_ce: 1; min_base_ce: 0.7983937; 
    FastEstimator-Train: step: 7000; base_ce: 0.7633059; steps/sec: 72.81; 
    FastEstimator-Train: step: 7000; epoch: 7; epoch_time: 13.74 sec; 
    FastEstimator-Eval: step: 7000; epoch: 7; base_ce: 0.80506575; clean_accuracy: 0.73; adversarial_accuracy: 0.2464; since_best_base_ce: 2; min_base_ce: 0.7983937; 
    FastEstimator-Train: step: 8000; base_ce: 0.57881784; steps/sec: 76.72; 
    FastEstimator-Train: step: 8000; epoch: 8; epoch_time: 13.03 sec; 
    FastEstimator-Eval: step: 8000; epoch: 8; base_ce: 0.8497379; clean_accuracy: 0.733; adversarial_accuracy: 0.214; since_best_base_ce: 3; min_base_ce: 0.7983937; 
    FastEstimator-Train: step: 9000; base_ce: 0.61386603; steps/sec: 78.1; 
    FastEstimator-Train: step: 9000; epoch: 9; epoch_time: 12.81 sec; 
    FastEstimator-Eval: step: 9000; epoch: 9; base_ce: 0.84185517; clean_accuracy: 0.731; adversarial_accuracy: 0.2206; since_best_base_ce: 4; min_base_ce: 0.7983937; 
    FastEstimator-Train: step: 10000; base_ce: 0.6447299; steps/sec: 77.0; 
    FastEstimator-Train: step: 10000; epoch: 10; epoch_time: 12.99 sec; 
    FastEstimator-Eval: step: 10000; epoch: 10; base_ce: 0.8941338; clean_accuracy: 0.7278; adversarial_accuracy: 0.1848; since_best_base_ce: 5; min_base_ce: 0.7983937; 
    FastEstimator-Finish: step: 10000; total_time: 148.17 sec; clean_model_lr: 0.001; 


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
    FastEstimator-Test: step: 10000; epoch: 10; clean_accuracy: 0.7178; adversarial_accuracy: 0.2674; 
    The whitebox FGSM attack reduced accuracy by 0.45
    -----------
    Adversarially Trained Network:
    FastEstimator-Test: step: 10000; epoch: 10; clean_accuracy: 0.6962; adversarial_accuracy: 0.4758; 
    The whitebox FGSM attack reduced accuracy by 0.22
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


    
![png](assets/branches/r1.0/example/adversarial_training/fgsm_files/fgsm_29_0.png)
    


As you can see, the adversarial images appear very similar to the unmodified images, and yet they are often able to modify the class predictions of the network. Note that if a network's prediction is already wrong, the attack is unlikely to change the incorrect prediction, but rather to increase the model's confidence in its incorrect prediction. 
