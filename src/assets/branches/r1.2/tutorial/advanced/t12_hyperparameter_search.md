# Advanced Tutorial 12: Hyperparameter Search

## Overview
In this tutorial, we will discuss the following topics:
* [FastEstimator Search API](tutorials/r1.2/advanced/t12_hyperparameter_search/#ta12searchapi)
    * [Getting the search results](tutorials/r1.2/advanced/t12_hyperparameter_search/#ta12searchresults)
    * [Saving and loading search results](tutorials/r1.2/advanced/t12_hyperparameter_search/#ta12saveload)
    * [Interruption-resilient search](tutorials/r1.2/advanced/t12_hyperparameter_search/#ta12interruption)
* [Example 1: Hyperparameter Tuning by Grid Search](tutorials/r1.2/advanced/t12_hyperparameter_search/#ta12example1)
* [Example 2: RUA Augmentation via Golden-Section Search](tutorials/r1.2/advanced/t12_hyperparameter_search/#ta12example2)

<a id='ta12searchapi'></a>

## Search API

There are many things in life that requires searching for an optimal solution in a given space, regardless of whether deep learning is involved. For example:
* what is the `x` that leads to the minimal value of `(x-3)**2`?
* what is the best `learning rate` and `batch size` combo that can produce the lowest evaluation loss after 2 epochs of training?
* what is the best augmentation magnitude that can lead to the highest evaluation accuracy?

The `fe.search` API is designed to make the search easier, the API can be used independently for any search problem, as it only requires the following two components:
1. objective function to measure the score of a solution.
2. whether a maximum or minimum score is desired.

We will start with a simple example using `Grid Search`. Say we want to find the `x` that produces the minimal value of `(x-3)**2`, where x is chosen from the list: `[0.5, 1.5, 2.9, 4, 5.3]`


```python
from fastestimator.search import GridSearch

def objective_fn(search_idx, x):
    return (x-3)**2

grid_search = GridSearch(score_fn=objective_fn, params={"x": [0.5, 1.5, 2.9, 4, 5.3]}, best_mode="min")
```

Note that in the score function, one of the arguments must be `search_idx`. This is to help user differentiate multiple search runs. To run the search, simply call:


```python
grid_search.fit()
```

    FastEstimator-Search: Evaluated {'x': 0.5, 'search_idx': 1}, score: 6.25
    FastEstimator-Search: Evaluated {'x': 1.5, 'search_idx': 2}, score: 2.25
    FastEstimator-Search: Evaluated {'x': 2.9, 'search_idx': 3}, score: 0.010000000000000018
    FastEstimator-Search: Evaluated {'x': 4, 'search_idx': 4}, score: 1
    FastEstimator-Search: Evaluated {'x': 5.3, 'search_idx': 5}, score: 5.289999999999999
    FastEstimator-Search: Grid Search Finished, best parameters: {'x': 2.9, 'search_idx': 3}, best score: 0.010000000000000018


<a id='ta12searchresults'></a>

### Getting the search results
After the search is done, you can also call the `search.get_best_results` or `search.get_search_results` to see the best and overall search history:


```python
print("best search result:")
print(grid_search.get_best_results())
```

    best search result:
    ({'x': 2.9, 'search_idx': 3}, 0.010000000000000018)



```python
print("search history:")
print(grid_search.get_search_results())
```

    search history:
    [({'x': 0.5, 'search_idx': 1}, 6.25), ({'x': 1.5, 'search_idx': 2}, 2.25), ({'x': 2.9, 'search_idx': 3}, 0.010000000000000018), ({'x': 4, 'search_idx': 4}, 1), ({'x': 5.3, 'search_idx': 5}, 5.289999999999999)]


<a id='ta12saveload'></a>

### Saving and loading search results

Once the search is done, you can also save the search results into the disk and later load them back using `save` and `load` methods:


```python
import tempfile
save_dir = tempfile.mkdtemp()

# save the state to save_dir
grid_search.save(save_dir) 

# instantiate a new object
grid_search2 = GridSearch(score_fn=objective_fn, params={"x": [0.5, 1.5, 2.9, 4, 5.3]}, best_mode="min") 

# load the previously saved state
grid_search2.load(save_dir)

# display the best result of the loaded instance
print(grid_search2.get_best_results()) 
```

    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmpb2geydkb/grid_search.json
    FastEstimator-Search: Loading the search state from /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmpb2geydkb/grid_search.json
    [{'x': 2.9, 'search_idx': 3}, 0.010000000000000018]


<a id='ta12interruption'></a>

### Interruption-resilient search
When you run search on a hardware that can be interrupted (like an AWS spot instance), you can provide a `save_dir` argument when calling `fit`. As a result, the search will automatically back up its result after each evaluation. Furthermore, when calling `fit` using the same `save_dir` the second time, it will first load the search results and then pick up from where it left off. 

To demonstrate this, we will use golden-section search on the same optimization problem. To simulate interruption, we will first iterate 10 times, then create a new instance and iterate another 10 times.


```python
from fastestimator.search import GoldenSection
save_dir2 = tempfile.mkdtemp()

gs_search =  GoldenSection(score_fn=objective_fn, x_min=0, x_max=6, max_iter=10, integer=False, best_mode="min")
gs_search.fit(save_dir=save_dir2)
```

    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 2.2917960675006306, 'search_idx': 1}, score: 0.5015528100075713
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 3.7082039324993694, 'search_idx': 2}, score: 0.5015528100075713
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 4.583592135001262, 'search_idx': 3}, score: 2.5077640500378555
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 3.1671842700025232, 'search_idx': 4}, score: 0.027950580136276586
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 2.832815729997476, 'search_idx': 5}, score: 0.027950580136276885
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 3.3738353924943216, 'search_idx': 6}, score: 0.1397529006813835
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 3.0394668524892743, 'search_idx': 7}, score: 0.0015576324454101358
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 2.9605331475107253, 'search_idx': 8}, score: 0.0015576324454101708
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 3.0882505650239747, 'search_idx': 9}, score: 0.00778816222705078
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 3.009316860045425, 'search_idx': 10}, score: 8.68038811060405e-05
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 2.9906831399545744, 'search_idx': 11}, score: 8.680388110604876e-05
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 3.0208331323984234, 'search_idx': 12}, score: 0.0004340194055302403
    FastEstimator-Search: Golden Section Search Finished, best parameters: {'x': 3.009316860045425, 'search_idx': 10}, best score: 8.68038811060405e-05


After interruption, we can create the instance and call `fit` on the same directory:


```python
gs_search2 =  GoldenSection(score_fn=objective_fn, x_min=0, x_max=6, max_iter=20, integer=False, best_mode="min")
gs_search2.fit(save_dir=save_dir2)
```

    FastEstimator-Search: Loading the search state from /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 3.002199412307572, 'search_idx': 13}, score: 4.8374144986998325e-06
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 2.997800587692428, 'search_idx': 14}, score: 4.8374144986998325e-06
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 3.0049180354302814, 'search_idx': 15}, score: 2.4187072493502697e-05
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 3.0005192108151366, 'search_idx': 16}, score: 2.695798705548303e-07
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 2.9994807891848634, 'search_idx': 17}, score: 2.695798705548303e-07
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 3.001160990677299, 'search_idx': 18}, score: 1.3478993527749865e-06
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 3.0001225690470252, 'search_idx': 19}, score: 1.502317128867399e-08
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 2.9998774309529748, 'search_idx': 20}, score: 1.502317128867399e-08
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 3.000274072721086, 'search_idx': 21}, score: 7.511585644356621e-08
    FastEstimator-Search: Saving the search state to /var/folders/cd/9k2rks597yl99yttyxmhkw7h0000gn/T/tmp5wp7bkir/golden_section_search.json
    FastEstimator-Search: Evaluated {'x': 3.0000289346270352, 'search_idx': 22}, score: 8.372126416682983e-10
    FastEstimator-Search: Golden Section Search Finished, best parameters: {'x': 3.0000289346270352, 'search_idx': 22}, best score: 8.372126416682983e-10


As we can see, the search started from search index 13 and proceeded for another 10 iterations.

<a id='ta12example1'></a>

## Example 1: Hyperparameter Tuning by Grid Search

In this example, we will use `GridSearch` on a real deep learning task to illustrate its usage. Specifically, given a batch size grid `[32, 64]` and learning rate grid `[1e-2 and 1e-3]`, we are interested in the optimial parameter that leads to the lowest test loss after 200 steps of training on MNIST dataset.


```python
import tensorflow as tf
import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp

def get_estimator(batch_size, lr):
    train_data, test_data = mnist.load_data()
    pipeline = fe.Pipeline(train_data=train_data,
                           test_data=test_data,
                           batch_size=batch_size,
                           ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])
    model = fe.build(model_fn=LeNet, optimizer_fn=lambda: tf.optimizers.Adam(lr))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=1,
                             max_train_steps_per_epoch=200)
    return estimator

def score_fn(search_idx, batch_size, lr):
    est = get_estimator(batch_size, lr)
    est.fit()
    hist = est.test(summary="myexp")
    test_loss = float(hist.history["test"]["ce"][200])
    return test_loss

mnist_grid_search = GridSearch(score_fn=score_fn, params={"batch_size": [32, 64], "lr": [1e-2, 1e-3]}, best_mode="min")
```


```python
mnist_grid_search.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; logging_interval: 100; num_device: 0;
    FastEstimator-Train: step: 1; ce: 2.292087;
    FastEstimator-Train: step: 100; ce: 0.3137861; steps/sec: 146.53;
    FastEstimator-Train: step: 200; ce: 0.31024012; steps/sec: 154.01;
    FastEstimator-Train: step: 200; epoch: 1; epoch_time: 2.9 sec;
    FastEstimator-Finish: step: 200; model_lr: 0.01; total_time: 3.35 sec;
    FastEstimator-Test: step: 200; epoch: 1; ce: 0.21014857;
    FastEstimator-Search: Evaluated {'batch_size': 32, 'lr': 0.01, 'search_idx': 1}, score: 0.21014857292175293
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; logging_interval: 100; num_device: 0;
    FastEstimator-Train: step: 1; ce: 2.3159466;
    FastEstimator-Train: step: 100; ce: 0.19143611; steps/sec: 137.74;
    FastEstimator-Train: step: 200; ce: 0.46275252; steps/sec: 164.92;
    FastEstimator-Train: step: 200; epoch: 1; epoch_time: 1.54 sec;
    FastEstimator-Finish: step: 200; model_lr: 0.001; total_time: 1.63 sec;
    FastEstimator-Test: step: 200; epoch: 1; ce: 0.21326771;
    FastEstimator-Search: Evaluated {'batch_size': 32, 'lr': 0.001, 'search_idx': 2}, score: 0.2132677137851715
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; logging_interval: 100; num_device: 0;
    FastEstimator-Train: step: 1; ce: 2.3091617;
    FastEstimator-Train: step: 100; ce: 0.26473606; steps/sec: 92.84;
    FastEstimator-Train: step: 200; ce: 0.094012745; steps/sec: 104.31;
    FastEstimator-Train: step: 200; epoch: 1; epoch_time: 2.26 sec;
    FastEstimator-Finish: step: 200; model_lr: 0.01; total_time: 2.36 sec;
    FastEstimator-Test: step: 200; epoch: 1; ce: 0.11212203;
    FastEstimator-Search: Evaluated {'batch_size': 64, 'lr': 0.01, 'search_idx': 3}, score: 0.11212202906608582
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; logging_interval: 100; num_device: 0;
    FastEstimator-Train: step: 1; ce: 2.2963362;
    FastEstimator-Train: step: 100; ce: 0.25377798; steps/sec: 95.01;
    FastEstimator-Train: step: 200; ce: 0.2405545; steps/sec: 113.96;
    FastEstimator-Train: step: 200; epoch: 1; epoch_time: 2.16 sec;
    FastEstimator-Finish: step: 200; model_lr: 0.001; total_time: 2.27 sec;
    FastEstimator-Test: step: 200; epoch: 1; ce: 0.16898704;
    FastEstimator-Search: Evaluated {'batch_size': 64, 'lr': 0.001, 'search_idx': 4}, score: 0.16898703575134277
    FastEstimator-Search: Grid Search Finished, best parameters: {'batch_size': 64, 'lr': 0.01, 'search_idx': 3}, best score: 0.11212202906608582


From the results we can see that, with only 200 steps of training, a bigger batch size and a larger learning rate combination is preferred.

<a id='ta12example2'></a>

## Example 2: RUA Augmentation via Golden-Section Search

In this example, we will use a built-in augmentation NumpyOp - RUA - and find the optimial level between 0 to 30 using `Golden-Section` search. The test result will be evaluated on the ciFAIR10 dataset after 500 steps of training.


```python
import tensorflow as tf
import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import cifair10
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax, RUA
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp

def get_estimator(level):
    train_data, test_data = cifair10.load_data()
    pipeline = fe.Pipeline(train_data=train_data,
                           test_data=test_data,
                           batch_size=64,
                           ops=[RUA(level=level, inputs="x", outputs="x", mode="train"), 
                                Minmax(inputs="x", outputs="x")])
    model = fe.build(model_fn=lambda: LeNet(input_shape=(32, 32, 3)), optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=1,
                             max_train_steps_per_epoch=500)
    return estimator

def score_fn(search_idx, level):
    est = get_estimator(level)
    est.fit()
    hist = est.test(summary="myexp")
    test_loss = float(hist.history["test"]["ce"][500])
    return test_loss

cifair10_gs_search = GoldenSection(score_fn=score_fn, x_min=0, x_max=30, max_iter=5, best_mode="min")
```


```python
cifair10_gs_search.fit()
```

    Downloading data from https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-10.zip
    168615936/168614301 [==============================] - 23s 0us/step
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; logging_interval: 100; num_device: 0;
    FastEstimator-Train: step: 1; ce: 2.299192;
    FastEstimator-Train: step: 100; ce: 1.9990821; steps/sec: 66.29;
    FastEstimator-Train: step: 200; ce: 1.7189915; steps/sec: 77.29;
    FastEstimator-Train: step: 300; ce: 1.3917459; steps/sec: 74.37;
    FastEstimator-Train: step: 400; ce: 1.7548736; steps/sec: 72.89;
    FastEstimator-Train: step: 500; ce: 1.6301897; steps/sec: 68.63;
    FastEstimator-Train: step: 500; epoch: 1; epoch_time: 7.26 sec;
    FastEstimator-Finish: step: 500; model_lr: 0.001; total_time: 7.39 sec;
    FastEstimator-Test: step: 500; epoch: 1; ce: 1.5230703;
    FastEstimator-Search: Evaluated {'level': 11, 'search_idx': 1}, score: 1.5230703353881836
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; logging_interval: 100; num_device: 0;
    FastEstimator-Train: step: 1; ce: 2.302093;
    FastEstimator-Train: step: 100; ce: 2.2276726; steps/sec: 59.68;
    FastEstimator-Train: step: 200; ce: 1.8196787; steps/sec: 67.21;
    FastEstimator-Train: step: 300; ce: 1.7125396; steps/sec: 67.33;
    FastEstimator-Train: step: 400; ce: 1.9558158; steps/sec: 61.51;
    FastEstimator-Train: step: 500; ce: 1.8046427; steps/sec: 57.66;
    FastEstimator-Train: step: 500; epoch: 1; epoch_time: 8.31 sec;
    FastEstimator-Finish: step: 500; model_lr: 0.001; total_time: 8.48 sec;
    FastEstimator-Test: step: 500; epoch: 1; ce: 1.5015539;
    FastEstimator-Search: Evaluated {'level': 18, 'search_idx': 2}, score: 1.5015538930892944
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; logging_interval: 100; num_device: 0;
    FastEstimator-Train: step: 1; ce: 2.2977169;
    FastEstimator-Train: step: 100; ce: 2.1189532; steps/sec: 57.86;
    FastEstimator-Train: step: 200; ce: 1.9732833; steps/sec: 63.68;
    FastEstimator-Train: step: 300; ce: 1.894378; steps/sec: 64.43;
    FastEstimator-Train: step: 400; ce: 1.8973417; steps/sec: 58.05;
    FastEstimator-Train: step: 500; ce: 1.9629371; steps/sec: 57.88;
    FastEstimator-Train: step: 500; epoch: 1; epoch_time: 8.65 sec;
    FastEstimator-Finish: step: 500; model_lr: 0.001; total_time: 8.82 sec;
    FastEstimator-Test: step: 500; epoch: 1; ce: 1.6366869;
    FastEstimator-Search: Evaluated {'level': 22, 'search_idx': 3}, score: 1.63668692111969
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; logging_interval: 100; num_device: 0;
    FastEstimator-Train: step: 1; ce: 2.3020601;
    FastEstimator-Train: step: 100; ce: 2.1016774; steps/sec: 58.07;
    FastEstimator-Train: step: 200; ce: 1.9502311; steps/sec: 62.04;
    FastEstimator-Train: step: 300; ce: 1.8442136; steps/sec: 62.1;
    FastEstimator-Train: step: 400; ce: 1.8126183; steps/sec: 56.72;
    FastEstimator-Train: step: 500; ce: 1.7455528; steps/sec: 53.78;
    FastEstimator-Train: step: 500; epoch: 1; epoch_time: 8.92 sec;
    FastEstimator-Finish: step: 500; model_lr: 0.001; total_time: 9.08 sec;
    FastEstimator-Test: step: 500; epoch: 1; ce: 1.5271283;
    FastEstimator-Search: Evaluated {'level': 15, 'search_idx': 4}, score: 1.5271283388137817
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; logging_interval: 100; num_device: 0;
    FastEstimator-Train: step: 1; ce: 2.324464;
    FastEstimator-Train: step: 100; ce: 2.2083392; steps/sec: 58.28;
    FastEstimator-Train: step: 200; ce: 2.0500524; steps/sec: 58.73;
    FastEstimator-Train: step: 300; ce: 1.8044301; steps/sec: 50.89;
    FastEstimator-Train: step: 400; ce: 2.0170183; steps/sec: 46.73;
    FastEstimator-Train: step: 500; ce: 1.8905652; steps/sec: 47.5;
    FastEstimator-Train: step: 500; epoch: 1; epoch_time: 9.99 sec;
    FastEstimator-Finish: step: 500; model_lr: 0.001; total_time: 10.17 sec;
    FastEstimator-Test: step: 500; epoch: 1; ce: 1.5466335;
    FastEstimator-Search: Evaluated {'level': 19, 'search_idx': 5}, score: 1.5466334819793701
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; logging_interval: 100; num_device: 0;
    FastEstimator-Train: step: 1; ce: 2.3080528;
    FastEstimator-Train: step: 100; ce: 2.210803; steps/sec: 57.77;
    FastEstimator-Train: step: 200; ce: 1.7868292; steps/sec: 61.79;
    FastEstimator-Train: step: 300; ce: 1.653916; steps/sec: 56.26;
    FastEstimator-Train: step: 400; ce: 1.8160346; steps/sec: 56.2;
    FastEstimator-Train: step: 500; ce: 1.74243; steps/sec: 55.42;
    FastEstimator-Train: step: 500; epoch: 1; epoch_time: 9.05 sec;
    FastEstimator-Finish: step: 500; model_lr: 0.001; total_time: 9.23 sec;
    FastEstimator-Test: step: 500; epoch: 1; ce: 1.5080028;
    FastEstimator-Search: Evaluated {'level': 16, 'search_idx': 6}, score: 1.508002758026123
        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 1; logging_interval: 100; num_device: 0;
    FastEstimator-Train: step: 1; ce: 2.302072;
    FastEstimator-Train: step: 100; ce: 2.0141592; steps/sec: 48.22;
    FastEstimator-Train: step: 200; ce: 1.8634024; steps/sec: 54.09;
    FastEstimator-Train: step: 300; ce: 1.7860389; steps/sec: 57.86;
    FastEstimator-Train: step: 400; ce: 1.7737244; steps/sec: 52.19;
    FastEstimator-Train: step: 500; ce: 1.5106875; steps/sec: 55.76;
    FastEstimator-Train: step: 500; epoch: 1; epoch_time: 9.78 sec;
    FastEstimator-Finish: step: 500; model_lr: 0.001; total_time: 10.01 sec;
    FastEstimator-Test: step: 500; epoch: 1; ce: 1.4974371;
    FastEstimator-Search: Evaluated {'level': 17, 'search_idx': 7}, score: 1.4974371194839478
    FastEstimator-Search: Golden Section Search Finished, best parameters: {'level': 17, 'search_idx': 7}, best score: 1.4974371194839478


In this example, the optimial level we found is 4. We can then train the model again using `level=4` to get the final model. In a real use case you will want to perform parameter search on a held-out evaluation set, and test the best parameters on the test set.
