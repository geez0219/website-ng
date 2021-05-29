# Advanced Tutorial 11: Model Calibration

## Overview
In this tutorial, we will discuss the following topics:
* [Calculating Calibration Error](./tutorials/r1.2/advanced/t11_model_calibration#ta11error)
* [Generating and Applying a Model Calibrator](./tutorials/r1.2/advanced/t11_model_calibration#ta11calibrator)

We'll start by getting the imports out of the way:


```python
import tempfile
import os

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.backend import squeeze, reduce_mean
from fastestimator.dataset.data import cifair10
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import CoarseDropout, Normalize, Calibrate
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.summary.logs import visualize_logs
from fastestimator.trace.adapt import PBMCalibrator
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import CalibrationError, MCC
from fastestimator.util import to_number, to_list

import matplotlib.pyplot as plt
import numpy as np

label_mapping = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}
```

And let's define a function to build a generic ciFAIR10 estimator. We will show how to use combinations of extra traces and post-processing ops to enhance this estimator throughout the tutorial.


```python
def build_estimator(extra_traces = None, postprocessing_ops = None):
    batch_size=128
    save_dir = tempfile.mkdtemp()
    extra_traces = to_list(extra_traces)
    postprocessing_ops = to_list(postprocessing_ops)
    train_data, eval_data = cifair10.load_data()
    test_data = eval_data.split(range(len(eval_data) // 2))
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        batch_size=batch_size,
        ops=[Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
             PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
             RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
             Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
             CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
             ],
        num_process=0)

    model = fe.build(model_fn=lambda: LeNet(input_shape=(32, 32, 3)), optimizer_fn="adam")
    network = fe.Network(
        ops=[
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=model, loss_name="ce")
        ], 
        pops=postprocessing_ops)  # <---- Some of the secret sauce will go here

    traces = [
        MCC(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="mcc", save_best_mode="max", load_best_final=True),
    ]
    traces = traces + extra_traces  # <---- Most of the secret sauce will go here
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=21,
                             traces=traces,
                             log_steps=300)
    return estimator
```

<a id='ta11error'></a>

## Calculating Calibration Error

Suppose you have a neural network that is performing image classification. For the sake of argument, let's imagine that the classification problem is to look at x-ray images and determine whether or not a patient has cancer. Let's further suppose that your model is very accurate: when it assigns a higher probability to 'cancer' the patient is almost always sick, and when it assigns a higher probability to 'healthy' the patient is almost always fine. It could be tempting to think that the job is done, but there is still a potential problem for real-world deployments of your model. Suppose a physician using your model runs an image and gets a report saying that it is 51% likely that the patient is healthy, and 49% likely that there is a cancerous tumor. In reality the patient is indeed healthy. From an accuracy point of view, your model is doing just fine. However, if the doctor sees that it is 49% likely that there is a tumor, they are likely to order a biopsy in order to be on the safe side. Taken to an extreme, suppose that your model always predicts a 49% probability of a tumor whenever it sees a healthy patient. Even though the model might have perfect accuracy, in practice it would always result in extra surgical procedures being performed. Ideally, if the model says that there is a 49% probability of a tumor, you would expect there to actually be a tumor in 49% of those cases. The discrepancy between a models predicted probability of a class and the true probability of that class conditioned on the prediction is measured as the calibration error. Calibration error is notoriously difficult to estimate correctly, but FE provides a `Trace` for this based on a [2019 NeurIPS spotlight paper](https://papers.nips.cc/paper/2019/file/f8c0c968632845cd133308b1a494967f-Paper.pdf) titled "Verified Uncertainty Calibration". 

The `CalibrationError` trace can be used just like any other metric trace, though it also optionally can compute confidence intervals around the estimated error. Keep in mind that to measure calibration error you would want your validation dataset to have a reasonable real-world class distribution (only a small percentage of people in the population actually have cancer, for example). For the purpose of easy illustration we will be using the ciFAIR10 dataset, and computing a 95% confidence interval for the estimated calibration error of the model:


```python
estimator = build_estimator(extra_traces=CalibrationError(true_key="y", pred_key="y_pred", confidence_interval=95))
```


```python
summary = estimator.fit("experiment1")
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; logging_interval: 300; num_device: 0;
    FastEstimator-Train: step: 1; ce: 2.3128014;
    FastEstimator-Train: step: 300; ce: 1.5598099; steps/sec: 24.91;
    FastEstimator-Train: step: 391; epoch: 1; epoch_time: 16.58 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpkuy03g7x/model_best_mcc.h5
    FastEstimator-Eval: step: 391; epoch: 1; calibration_error: (0.0297, 0.0334, 0.0376); ce: 1.3315556; max_mcc: 0.46224125252635423; mcc: 0.46224125252635423; since_best_mcc: 0;
    FastEstimator-Train: step: 600; ce: 1.3069022; steps/sec: 24.27;
    FastEstimator-Train: step: 782; epoch: 2; epoch_time: 16.05 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpkuy03g7x/model_best_mcc.h5
    FastEstimator-Eval: step: 782; epoch: 2; calibration_error: (0.0427, 0.0476, 0.0517); ce: 1.1747007; max_mcc: 0.534939578707041; mcc: 0.534939578707041; since_best_mcc: 0;
    FastEstimator-Train: step: 900; ce: 1.0707315; steps/sec: 24.28;
    FastEstimator-Train: step: 1173; epoch: 3; epoch_time: 16.27 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpkuy03g7x/model_best_mcc.h5
    FastEstimator-Eval: step: 1173; epoch: 3; calibration_error: (0.0318, 0.0364, 0.0417); ce: 1.0508112; max_mcc: 0.5908518626163255; mcc: 0.5908518626163255; since_best_mcc: 0;
    FastEstimator-Train: step: 1200; ce: 1.0858493; steps/sec: 23.92;
    FastEstimator-Train: step: 1500; ce: 1.1012355; steps/sec: 23.68;
    FastEstimator-Train: step: 1564; epoch: 4; epoch_time: 16.62 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpkuy03g7x/model_best_mcc.h5
    FastEstimator-Eval: step: 1564; epoch: 4; calibration_error: (0.0312, 0.0358, 0.04); ce: 0.9772445; max_mcc: 0.6151001297640063; mcc: 0.6151001297640063; since_best_mcc: 0;
    FastEstimator-Train: step: 1800; ce: 1.0599184; steps/sec: 23.18;
    FastEstimator-Train: step: 1955; epoch: 5; epoch_time: 16.8 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpkuy03g7x/model_best_mcc.h5
    FastEstimator-Eval: step: 1955; epoch: 5; calibration_error: (0.0257, 0.0331, 0.0391); ce: 0.89843607; max_mcc: 0.6503338759880644; mcc: 0.6503338759880644; since_best_mcc: 0;
    FastEstimator-Train: step: 2100; ce: 0.95765936; steps/sec: 23.39;
    FastEstimator-Train: step: 2346; epoch: 6; epoch_time: 16.81 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpkuy03g7x/model_best_mcc.h5
    FastEstimator-Eval: step: 2346; epoch: 6; calibration_error: (0.0255, 0.0298, 0.0338); ce: 0.8587847; max_mcc: 0.6642532225424916; mcc: 0.6642532225424916; since_best_mcc: 0;
    FastEstimator-Train: step: 2400; ce: 0.9164876; steps/sec: 22.98;
    FastEstimator-Train: step: 2700; ce: 0.9614507; steps/sec: 22.95;
    FastEstimator-Train: step: 2737; epoch: 7; epoch_time: 17.08 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpkuy03g7x/model_best_mcc.h5
    FastEstimator-Eval: step: 2737; epoch: 7; calibration_error: (0.0263, 0.031, 0.0368); ce: 0.8350588; max_mcc: 0.6781491163974881; mcc: 0.6781491163974881; since_best_mcc: 0;
    FastEstimator-Train: step: 3000; ce: 0.85875076; steps/sec: 22.88;
    FastEstimator-Train: step: 3128; epoch: 8; epoch_time: 17.08 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpkuy03g7x/model_best_mcc.h5
    FastEstimator-Eval: step: 3128; epoch: 8; calibration_error: (0.0175, 0.0234, 0.0271); ce: 0.7929346; max_mcc: 0.6882606610662204; mcc: 0.6882606610662204; since_best_mcc: 0;
    FastEstimator-Train: step: 3300; ce: 0.8300022; steps/sec: 22.92;
    FastEstimator-Train: step: 3519; epoch: 9; epoch_time: 17.21 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpkuy03g7x/model_best_mcc.h5
    FastEstimator-Eval: step: 3519; epoch: 9; calibration_error: (0.0144, 0.0206, 0.0264); ce: 0.7876051; max_mcc: 0.6927864379192522; mcc: 0.6927864379192522; since_best_mcc: 0;
    FastEstimator-Train: step: 3600; ce: 0.85931337; steps/sec: 22.65;
    FastEstimator-Train: step: 3900; ce: 0.81882757; steps/sec: 22.8;
    FastEstimator-Train: step: 3910; epoch: 10; epoch_time: 17.15 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpkuy03g7x/model_best_mcc.h5
    FastEstimator-Eval: step: 3910; epoch: 10; calibration_error: (0.0209, 0.0243, 0.0282); ce: 0.76851153; max_mcc: 0.7093341814909904; mcc: 0.7093341814909904; since_best_mcc: 0;
    FastEstimator-Train: step: 4200; ce: 0.82108325; steps/sec: 22.96;
    FastEstimator-Train: step: 4301; epoch: 11; epoch_time: 16.95 sec;
    FastEstimator-Eval: step: 4301; epoch: 11; calibration_error: (0.0215, 0.0288, 0.0343); ce: 0.74303454; max_mcc: 0.7093341814909904; mcc: 0.707637570222403; since_best_mcc: 1;
    FastEstimator-Train: step: 4500; ce: 1.1019704; steps/sec: 22.92;
    FastEstimator-Train: step: 4692; epoch: 12; epoch_time: 17.18 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpkuy03g7x/model_best_mcc.h5
    FastEstimator-Eval: step: 4692; epoch: 12; calibration_error: (0.0217, 0.0262, 0.0301); ce: 0.73569554; max_mcc: 0.7152104877875541; mcc: 0.7152104877875541; since_best_mcc: 0;
    FastEstimator-Train: step: 4800; ce: 0.8793318; steps/sec: 22.58;
    FastEstimator-Train: step: 5083; epoch: 13; epoch_time: 17.2 sec;
    FastEstimator-Eval: step: 5083; epoch: 13; calibration_error: (0.0282, 0.0319, 0.0382); ce: 0.7367064; max_mcc: 0.7152104877875541; mcc: 0.7065484032881655; since_best_mcc: 1;
    FastEstimator-Train: step: 5100; ce: 0.7968985; steps/sec: 22.95;
    FastEstimator-Train: step: 5400; ce: 0.80106986; steps/sec: 22.54;
    FastEstimator-Train: step: 5474; epoch: 14; epoch_time: 17.3 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpkuy03g7x/model_best_mcc.h5
    FastEstimator-Eval: step: 5474; epoch: 14; calibration_error: (0.017, 0.0215, 0.0258); ce: 0.69848; max_mcc: 0.731208053521974; mcc: 0.731208053521974; since_best_mcc: 0;
    FastEstimator-Train: step: 5700; ce: 0.7631496; steps/sec: 22.18;
    FastEstimator-Train: step: 5865; epoch: 15; epoch_time: 17.58 sec;
    FastEstimator-Eval: step: 5865; epoch: 15; calibration_error: (0.0291, 0.0338, 0.037); ce: 0.729518; max_mcc: 0.731208053521974; mcc: 0.713921402106124; since_best_mcc: 1;
    FastEstimator-Train: step: 6000; ce: 0.7238472; steps/sec: 22.29;
    FastEstimator-Train: step: 6256; epoch: 16; epoch_time: 17.46 sec;
    FastEstimator-Eval: step: 6256; epoch: 16; calibration_error: (0.0232, 0.0277, 0.032); ce: 0.70096886; max_mcc: 0.731208053521974; mcc: 0.7296463555135289; since_best_mcc: 2;
    FastEstimator-Train: step: 6300; ce: 0.76086265; steps/sec: 22.57;
    FastEstimator-Train: step: 6600; ce: 0.7575444; steps/sec: 23.06;
    FastEstimator-Train: step: 6647; epoch: 17; epoch_time: 17.06 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpkuy03g7x/model_best_mcc.h5
    FastEstimator-Eval: step: 6647; epoch: 17; calibration_error: (0.0181, 0.0232, 0.0297); ce: 0.68079054; max_mcc: 0.7398210967101186; mcc: 0.7398210967101186; since_best_mcc: 0;
    FastEstimator-Train: step: 6900; ce: 0.81247497; steps/sec: 22.66;
    FastEstimator-Train: step: 7038; epoch: 18; epoch_time: 17.24 sec;
    FastEstimator-Eval: step: 7038; epoch: 18; calibration_error: (0.0187, 0.0218, 0.0281); ce: 0.6772504; max_mcc: 0.7398210967101186; mcc: 0.7364544417648297; since_best_mcc: 1;
    FastEstimator-Train: step: 7200; ce: 0.6884398; steps/sec: 22.48;
    FastEstimator-Train: step: 7429; epoch: 19; epoch_time: 17.38 sec;
    FastEstimator-Eval: step: 7429; epoch: 19; calibration_error: (0.0355, 0.0411, 0.0459); ce: 0.71631813; max_mcc: 0.7398210967101186; mcc: 0.7219233509501778; since_best_mcc: 2;
    FastEstimator-Train: step: 7500; ce: 0.99218196; steps/sec: 22.69;
    FastEstimator-Train: step: 7800; ce: 0.811993; steps/sec: 22.69;
    FastEstimator-Train: step: 7820; epoch: 20; epoch_time: 17.23 sec;
    FastEstimator-Eval: step: 7820; epoch: 20; calibration_error: (0.0238, 0.028, 0.0331); ce: 0.67875355; max_mcc: 0.7398210967101186; mcc: 0.7334906886301213; since_best_mcc: 3;
    FastEstimator-Train: step: 8100; ce: 0.8382884; steps/sec: 22.38;
    FastEstimator-Train: step: 8211; epoch: 21; epoch_time: 17.52 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpkuy03g7x/model_best_mcc.h5
    FastEstimator-Eval: step: 8211; epoch: 21; calibration_error: (0.0213, 0.0269, 0.0306); ce: 0.652671; max_mcc: 0.7496558877385842; mcc: 0.7496558877385842; since_best_mcc: 0;
    FastEstimator-BestModelSaver: Restoring model from /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpkuy03g7x/model_best_mcc.h5
    FastEstimator-Finish: step: 8211; model_lr: 0.001; total_time: 559.65 sec;



```python
estimator.test()
```

    FastEstimator-Test: step: 8211; epoch: 21; calibration_error: (0.0226, 0.0279, 0.0345); ce: 0.6998544; mcc: 0.7373376975856221;





    <fastestimator.summary.summary.Summary at 0x1b6fd3668>



Let's take a look at how the calibration error changed over training:


```python
visualize_logs([summary], include_metrics={'calibration_error', 'mcc', 'ce'})
```


    
![png](assets/branches/r1.2/tutorial/advanced/t11_model_calibration_files/t11_model_calibration_11_0.png)
    


As we can see from the graph above, calibration error is significantly more noisy than classical metrics like mcc or accuracy. In this case it does seem to have improved somewhat with training, though the correlation isn't strong enough to expect to be able to eliminate your calibration error just by training longer. Instead, we will see how you can effectively calibrate a model after-the-fact:

<a id='ta11calibrator'></a>

## Generating and Applying a Model Calibrator

While there have been many proposed approaches for model calibration, we will again be leveraging the Verified Uncertainty Calibration paper mentioned above to achieve highly sample-efficient model re-calibration. There are two steps involved here. The first step is that we will use the `PBMCalibrator` trace to generate a 'platt binner marginal calibrator'. This calibrator is separate from the neural network, but will take neural network outputs and return calibrated outputs. A consequence of performing this calibration is that the output vector for a prediction will no longer sum to 1, since each class is calibrated independently. 

Of course, simply having such a calibration object is not useful if we don't use it. To make use of our calibrator object we will use the `Calibrate` numpyOp, which can load any calibrator object from disk and then apply it during `Network` post-processing. Since we are using a best model saver, we will only save the calibrator object when our since_best is 0 so that when we re-load the best model we will also be loading the correct calibrator for that model.  


```python
save_path = os.path.join(tempfile.mkdtemp(), 'calibrator.pkl')
estimator = build_estimator(extra_traces=[CalibrationError(true_key="y", pred_key="y_pred", confidence_interval=95), 
                                          PBMCalibrator(true_key="y", pred_key="y_pred", save_path=save_path, save_if_key="since_best_mcc", mode="eval"),
                                          # We will also compare the MCC and calibration error between the original and calibrated samples:
                                          MCC(true_key="y", pred_key="y_pred_calibrated", output_name="mcc (calibrated)", mode="test"),
                                          CalibrationError(true_key="y", pred_key="y_pred_calibrated", output_name="calibration_error (calibrated)", confidence_interval=95, mode="test"), 
                                          ],
                           postprocessing_ops = Calibrate(inputs="y_pred", outputs="y_pred_calibrated", calibration_fn=save_path, mode="test"))
```


```python
summary = estimator.fit("experiment2")
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; logging_interval: 300; num_device: 0;
    WARNING:tensorflow:5 out of the last 43 calls to <function TFNetwork._forward_step_static at 0x1b7938e18> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    FastEstimator-Train: step: 1; ce: 2.3104477;
    FastEstimator-Train: step: 300; ce: 1.4325299; steps/sec: 22.16;
    FastEstimator-Train: step: 391; epoch: 1; epoch_time: 18.26 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmppvkulhpx/model1_best_mcc.h5
    FastEstimator-PBMCalibrator: Calibrator written to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpsgbutr41/calibrator.pkl
    FastEstimator-Eval: step: 391; epoch: 1; calibration_error: (0.0452, 0.0497, 0.0537); ce: 1.3418019; max_mcc: 0.471251731557843; mcc: 0.471251731557843; since_best_mcc: 0;
    FastEstimator-Train: step: 600; ce: 1.4104708; steps/sec: 20.82;
    FastEstimator-Train: step: 782; epoch: 2; epoch_time: 18.5 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmppvkulhpx/model1_best_mcc.h5
    FastEstimator-PBMCalibrator: Calibrator written to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpsgbutr41/calibrator.pkl
    FastEstimator-Eval: step: 782; epoch: 2; calibration_error: (0.035, 0.039, 0.0452); ce: 1.1203645; max_mcc: 0.5493889475603118; mcc: 0.5493889475603118; since_best_mcc: 0;
    FastEstimator-Train: step: 900; ce: 1.104803; steps/sec: 21.38;
    FastEstimator-Train: step: 1173; epoch: 3; epoch_time: 18.27 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmppvkulhpx/model1_best_mcc.h5
    FastEstimator-PBMCalibrator: Calibrator written to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpsgbutr41/calibrator.pkl
    FastEstimator-Eval: step: 1173; epoch: 3; calibration_error: (0.0191, 0.0228, 0.0272); ce: 1.024723; max_mcc: 0.6019568874281185; mcc: 0.6019568874281185; since_best_mcc: 0;
    FastEstimator-Train: step: 1200; ce: 1.1308318; steps/sec: 21.26;
    FastEstimator-Train: step: 1500; ce: 1.0688322; steps/sec: 21.58;
    FastEstimator-Train: step: 1564; epoch: 4; epoch_time: 18.34 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmppvkulhpx/model1_best_mcc.h5
    FastEstimator-PBMCalibrator: Calibrator written to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpsgbutr41/calibrator.pkl
    FastEstimator-Eval: step: 1564; epoch: 4; calibration_error: (0.0339, 0.0373, 0.0418); ce: 0.97765255; max_mcc: 0.62197137268699; mcc: 0.62197137268699; since_best_mcc: 0;
    FastEstimator-Train: step: 1800; ce: 1.1226903; steps/sec: 20.74;
    FastEstimator-Train: step: 1955; epoch: 5; epoch_time: 18.6 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmppvkulhpx/model1_best_mcc.h5
    FastEstimator-PBMCalibrator: Calibrator written to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpsgbutr41/calibrator.pkl
    FastEstimator-Eval: step: 1955; epoch: 5; calibration_error: (0.0324, 0.0367, 0.0402); ce: 0.9452076; max_mcc: 0.6421886616786081; mcc: 0.6421886616786081; since_best_mcc: 0;
    FastEstimator-Train: step: 2100; ce: 1.1943399; steps/sec: 21.28;
    FastEstimator-Train: step: 2346; epoch: 6; epoch_time: 17.92 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmppvkulhpx/model1_best_mcc.h5
    FastEstimator-PBMCalibrator: Calibrator written to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpsgbutr41/calibrator.pkl
    FastEstimator-Eval: step: 2346; epoch: 6; calibration_error: (0.0247, 0.0299, 0.0356); ce: 0.87151146; max_mcc: 0.6599993209174118; mcc: 0.6599993209174118; since_best_mcc: 0;
    FastEstimator-Train: step: 2400; ce: 0.96866685; steps/sec: 22.06;
    FastEstimator-Train: step: 2700; ce: 0.8802682; steps/sec: 21.65;
    FastEstimator-Train: step: 2737; epoch: 7; epoch_time: 18.02 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmppvkulhpx/model1_best_mcc.h5
    FastEstimator-PBMCalibrator: Calibrator written to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpsgbutr41/calibrator.pkl
    FastEstimator-Eval: step: 2737; epoch: 7; calibration_error: (0.0381, 0.0429, 0.0482); ce: 0.85621566; max_mcc: 0.6663392798589082; mcc: 0.6663392798589082; since_best_mcc: 0;
    FastEstimator-Train: step: 3000; ce: 0.7641719; steps/sec: 21.8;
    FastEstimator-Train: step: 3128; epoch: 8; epoch_time: 18.36 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmppvkulhpx/model1_best_mcc.h5
    FastEstimator-PBMCalibrator: Calibrator written to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpsgbutr41/calibrator.pkl
    FastEstimator-Eval: step: 3128; epoch: 8; calibration_error: (0.0325, 0.037, 0.043); ce: 0.8335171; max_mcc: 0.6738276146740207; mcc: 0.6738276146740207; since_best_mcc: 0;
    FastEstimator-Train: step: 3300; ce: 1.0184993; steps/sec: 20.26;
    FastEstimator-Train: step: 3519; epoch: 9; epoch_time: 18.79 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmppvkulhpx/model1_best_mcc.h5
    FastEstimator-PBMCalibrator: Calibrator written to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpsgbutr41/calibrator.pkl
    FastEstimator-Eval: step: 3519; epoch: 9; calibration_error: (0.0155, 0.0191, 0.0234); ce: 0.77648324; max_mcc: 0.7027027758799406; mcc: 0.7027027758799406; since_best_mcc: 0;
    FastEstimator-Train: step: 3600; ce: 0.92807305; steps/sec: 20.88;
    FastEstimator-Train: step: 3900; ce: 0.88444686; steps/sec: 21.79;
    FastEstimator-Train: step: 3910; epoch: 10; epoch_time: 18.28 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmppvkulhpx/model1_best_mcc.h5
    FastEstimator-PBMCalibrator: Calibrator written to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpsgbutr41/calibrator.pkl
    FastEstimator-Eval: step: 3910; epoch: 10; calibration_error: (0.0224, 0.0264, 0.0307); ce: 0.7616478; max_mcc: 0.7069650546236691; mcc: 0.7069650546236691; since_best_mcc: 0;
    FastEstimator-Train: step: 4200; ce: 0.8382786; steps/sec: 22.39;
    FastEstimator-Train: step: 4301; epoch: 11; epoch_time: 17.44 sec;
    FastEstimator-Eval: step: 4301; epoch: 11; calibration_error: (0.0279, 0.0327, 0.0384); ce: 0.7522718; max_mcc: 0.7069650546236691; mcc: 0.6990289129766644; since_best_mcc: 1;
    FastEstimator-Train: step: 4500; ce: 0.8221933; steps/sec: 22.34;
    FastEstimator-Train: step: 4692; epoch: 12; epoch_time: 17.6 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmppvkulhpx/model1_best_mcc.h5
    FastEstimator-PBMCalibrator: Calibrator written to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpsgbutr41/calibrator.pkl
    FastEstimator-Eval: step: 4692; epoch: 12; calibration_error: (0.0215, 0.0269, 0.0313); ce: 0.74621546; max_mcc: 0.7195339120387272; mcc: 0.7195339120387272; since_best_mcc: 0;
    FastEstimator-Train: step: 4800; ce: 0.9214088; steps/sec: 22.0;
    FastEstimator-Train: step: 5083; epoch: 13; epoch_time: 17.89 sec;
    FastEstimator-Eval: step: 5083; epoch: 13; calibration_error: (0.0336, 0.038, 0.043); ce: 0.78903353; max_mcc: 0.7195339120387272; mcc: 0.6991651579922098; since_best_mcc: 1;
    FastEstimator-Train: step: 5100; ce: 1.0087302; steps/sec: 21.96;
    FastEstimator-Train: step: 5400; ce: 0.8384258; steps/sec: 21.53;
    FastEstimator-Train: step: 5474; epoch: 14; epoch_time: 18.02 sec;
    FastEstimator-Eval: step: 5474; epoch: 14; calibration_error: (0.0345, 0.0391, 0.0438); ce: 0.7304886; max_mcc: 0.7195339120387272; mcc: 0.7168369900325905; since_best_mcc: 2;
    FastEstimator-Train: step: 5700; ce: 0.89664674; steps/sec: 22.06;
    FastEstimator-Train: step: 5865; epoch: 15; epoch_time: 17.62 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmppvkulhpx/model1_best_mcc.h5
    FastEstimator-PBMCalibrator: Calibrator written to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpsgbutr41/calibrator.pkl
    FastEstimator-Eval: step: 5865; epoch: 15; calibration_error: (0.0172, 0.0218, 0.0294); ce: 0.7002689; max_mcc: 0.7310497448441268; mcc: 0.7310497448441268; since_best_mcc: 0;
    FastEstimator-Train: step: 6000; ce: 0.75543165; steps/sec: 22.17;
    FastEstimator-Train: step: 6256; epoch: 16; epoch_time: 17.65 sec;
    FastEstimator-Eval: step: 6256; epoch: 16; calibration_error: (0.0192, 0.0243, 0.0286); ce: 0.6951452; max_mcc: 0.7310497448441268; mcc: 0.7219706354699861; since_best_mcc: 1;
    FastEstimator-Train: step: 6300; ce: 0.8381767; steps/sec: 22.37;
    FastEstimator-Train: step: 6600; ce: 0.71868813; steps/sec: 22.08;
    FastEstimator-Train: step: 6647; epoch: 17; epoch_time: 17.64 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmppvkulhpx/model1_best_mcc.h5
    FastEstimator-PBMCalibrator: Calibrator written to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpsgbutr41/calibrator.pkl
    FastEstimator-Eval: step: 6647; epoch: 17; calibration_error: (0.0165, 0.0201, 0.0247); ce: 0.6688763; max_mcc: 0.7387916691470418; mcc: 0.7387916691470418; since_best_mcc: 0;
    FastEstimator-Train: step: 6900; ce: 0.75006866; steps/sec: 22.05;
    FastEstimator-Train: step: 7038; epoch: 18; epoch_time: 17.67 sec;
    FastEstimator-Eval: step: 7038; epoch: 18; calibration_error: (0.0326, 0.0354, 0.0395); ce: 0.70194936; max_mcc: 0.7387916691470418; mcc: 0.7251760610978063; since_best_mcc: 1;
    FastEstimator-Train: step: 7200; ce: 0.7030199; steps/sec: 22.46;
    FastEstimator-Train: step: 7429; epoch: 19; epoch_time: 17.64 sec;
    FastEstimator-Eval: step: 7429; epoch: 19; calibration_error: (0.0297, 0.0354, 0.0402); ce: 0.691007; max_mcc: 0.7387916691470418; mcc: 0.7330375037002735; since_best_mcc: 2;
    FastEstimator-Train: step: 7500; ce: 0.77258766; steps/sec: 22.06;
    FastEstimator-Train: step: 7800; ce: 0.697283; steps/sec: 22.1;
    FastEstimator-Train: step: 7820; epoch: 20; epoch_time: 17.56 sec;
    FastEstimator-BestModelSaver: Saved model to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmppvkulhpx/model1_best_mcc.h5
    FastEstimator-PBMCalibrator: Calibrator written to /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpsgbutr41/calibrator.pkl
    FastEstimator-Eval: step: 7820; epoch: 20; calibration_error: (0.0166, 0.0209, 0.0279); ce: 0.66420925; max_mcc: 0.7462139868420128; mcc: 0.7462139868420128; since_best_mcc: 0;
    FastEstimator-Train: step: 8100; ce: 0.7611714; steps/sec: 22.27;
    FastEstimator-Train: step: 8211; epoch: 21; epoch_time: 17.52 sec;
    FastEstimator-Eval: step: 8211; epoch: 21; calibration_error: (0.0225, 0.0282, 0.0333); ce: 0.6680258; max_mcc: 0.7462139868420128; mcc: 0.7454059655301143; since_best_mcc: 1;
    FastEstimator-BestModelSaver: Restoring model from /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmppvkulhpx/model1_best_mcc.h5
    FastEstimator-Finish: step: 8211; model1_lr: 0.001; total_time: 587.74 sec;



```python
estimator.test()
```

    FastEstimator-Calibrate: calibration function loaded from /var/folders/lx/drkxftt117gblvgsp1p39rlc0000gn/T/tmpsgbutr41/calibrator.pkl
    FastEstimator-Test: step: 8211; epoch: 21; calibration_error: (0.0204, 0.0242, 0.03); calibration_error (calibrated): (0.0012, 0.0047, 0.0088); ce: 0.7153029; mcc: 0.7339172532443659; mcc (calibrated): 0.7296598230150076;





    <fastestimator.summary.summary.Summary at 0x1b9690f28>




```python
visualize_logs([summary], include_metrics={'calibration_error', 'mcc', 'ce', "calibration_error (calibrated)", "mcc (calibrated)"})
```


    
![png](assets/branches/r1.2/tutorial/advanced/t11_model_calibration_files/t11_model_calibration_18_0.png)
    



```python
delta = summary.history['test']['mcc (calibrated)'][8211] - summary.history['test']['mcc'][8211]
relative_delta = delta / summary.history['test']['mcc'][8211]
print(f"mcc change after calibration: {delta} ({relative_delta*100}%)")
```

    mcc change after calibration: -0.004257430229358206 (-0.5800967630257695%)



```python
delta = summary.history['test']['calibration_error (calibrated)'][8211].y - summary.history['test']['calibration_error'][8211].y
relative_delta = delta / summary.history['test']['calibration_error'][8211].y
print(f"calibration error change after calibration: {delta} ({relative_delta*100}%)")
```

    calibration error change after calibration: -0.0195 (-80.57851239669421%)


As we can see from the graphs and values above, with the use of a platt binning marginal calibrator we can dramatically reduce a model's calibration error (in this case by over 80%) while sacrificing only a very small amount of model performance (in this case less than a 1% reduction in MCC). 
