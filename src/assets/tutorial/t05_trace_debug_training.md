# Tutorial 5: Trace - training control and debugging

In this tutorial, we will talk about another important concept in FastEstimator - __Trace__. It will mainly be used to control training and for debugging.

`Trace` is a class, which contains 6 event functions listed below. Each event function will be executed on different events of the training loop when the user adds `Trace` inside `Estimator`. If you are a Keras user, you can think of `Trace` as a combination of callbacks and metrics. 
* on_begin
* on_epoch_begin
* on_batch_begin
* on_batch_end
* on_epoch_end
* on_end

`Trace` differs from keras's callback in the following places:
1. Trace has full access to the preprocessing data and prediction data
2. Different traces can pass data among each other
3. Trace is simpler and has fewer event functions than keras callbacks

`Trace` can be used for anything that involves training loop, such as changing learning rate, calculating metrics, writing checkpoints...


```python
# Import libraries
import tempfile
import numpy as np
import tensorflow as tf
import fastestimator as fe
```

## Using Trace to debug training loop

Since `Trace` can have full access to data used in training loop, one natural usage of `Trace` is debugging training loop, for example, printing network prediction for each batch.

Remember in tutorial 3, we customized an operation that scales the prediction score by 10 and writes a new key, let's see whether the operation is working correctly using `Trace`.

### 1) Define the operation to test,  pipeline and network.


```python
from fastestimator.architecture import LeNet
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.op.tensorop.loss import SparseCategoricalCrossentropy
from fastestimator.op.tensorop import Minmax
from fastestimator.op import TensorOp

# We define the scaling operation.
class Scale(TensorOp):
    def forward(self, data, state):
        data = data * 10
        return data

# We load data, create dictionnaries and prepare the Pipeline.
(x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
train_data = {"x": np.expand_dims(x_train, -1), "y": y_train}
eval_data = {"x": np.expand_dims(x_eval, -1), "y": y_eval}
data = {"train": train_data, "eval": eval_data}
pipeline = fe.Pipeline(batch_size=32, data=data, ops=Minmax(inputs="x", outputs="x"))

# We prepare the model and network, which will use the scaling operation.
model = fe.build(model_def=LeNet, model_name="lenet", optimizer="adam", loss_name="loss")
network = fe.Network(
    ops=[ModelOp(inputs="x", model=model, outputs="y_pred"), 
         SparseCategoricalCrossentropy(inputs=("y", "y_pred"),outputs="loss"), 
         Scale(inputs="y_pred", outputs="y_pred_scaled")])
```

### 2) Define the trace
We want to display, at the end of each batch during training, the scaled prediction.

We can access the batch_data with `state["batch"]` and then print the information we want to check, here:
- the step ("batch_idx")
- keys of the data (what is contained in each batch data: y, y_pred, y_pred_scaled, loss...)
- scaled prediction ("y_pred_scaled").


```python
from fastestimator.trace import Trace
from fastestimator.trace import Accuracy, ModelSaver

# We define a trace to show the predictions and test the scaling op.
class ShowPred(Trace):

    def on_batch_end(self, state): # We only want to show predictions at the end of the batch
        if state["mode"] == "train": # and only during training
            batch_data = state["batch"] 
            print("step: {}".format(state["batch_idx"]))
            print("batch data has following keys: {}".format(list(batch_data.keys())))
            print("scaled_prediction is:")
            print(batch_data["y_pred_scaled"])

# We finally define the estimator, specifying the trace argument. For debugging, we only use one epoch with one step.
estimator = fe.Estimator(network=network, pipeline=pipeline, epochs=1, traces=ShowPred(), steps_per_epoch=1)
```


```python
# We launch the training and can see what the scaled prediction looks like.
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 0; total_train_steps: 1; lenet_lr: 0.001; 
    step: 0
    batch data has following keys: ['x', 'y_pred_scaled', 'y', 'y_pred', 'loss']
    scaled_prediction is:
    tf.Tensor(
    [[0.96793294 0.9402648  0.93156326 0.9278286  0.9109819  0.9401245
      1.0638573  1.0010896  1.091351   1.2250059 ]
     [1.038612   0.9753925  0.9636226  0.9484253  0.89349127 0.95113695
      0.98204666 0.99905026 1.0780157  1.1702076 ]
     [1.029746   0.95729184 0.8991195  1.004261   0.90899205 0.96533054
      1.0602818  0.99144405 1.0408678  1.142666  ]
     [0.97642314 0.9921119  0.9373445  0.9310066  0.9308049  0.9860778
      1.0083483  0.98910564 1.1083883  1.1403897 ]
     [0.98148113 0.97943723 0.961184   0.89693433 0.898618   0.9587736
      1.0280821  0.9779144  1.1194493  1.1981252 ]
     [1.0126423  0.92431533 1.0063522  1.0090388  0.9094488  0.9768225
      1.0436704  0.9719753  1.0500734  1.0956608 ]
     [1.0654325  0.93496376 0.9329829  0.92262924 0.93740374 0.92491126
      1.0189292  0.9860127  1.1034629  1.1732719 ]
     [1.0238682  0.9843135  0.9675458  0.93035495 0.9180299  0.98023283
      1.0108459  0.9739873  1.0772535  1.1335682 ]
     [0.995144   0.9350034  0.9910743  0.9604698  0.9369736  0.9951952
      1.0367818  0.9618515  1.0682566  1.1192503 ]
     [1.0057276  0.9793299  1.0069429  0.9844189  0.8872866  0.9650487
      1.0052571  1.004198   1.0567607  1.1050293 ]
     [1.0207938  0.94707257 0.9670769  0.88475394 0.9313483  0.95185614
      1.0151181  0.9960452  1.0897725  1.196162  ]
     [1.017458   0.97480786 0.9615949  0.9348714  0.9275124  0.9366972
      0.9979079  0.96881557 1.0895962  1.1907394 ]
     [0.9560565  0.9573864  0.9379517  0.9201845  0.86426914 0.9550128
      1.015967   1.0171957  1.1162956  1.2596809 ]
     [0.938841   0.9503144  0.98008484 0.8984654  0.8628151  0.97239584
      1.0499079  1.0336926  1.0848446  1.2286388 ]
     [1.0136739  0.95339763 0.92607105 0.94601405 0.8835093  0.94986665
      1.0498883  1.0067003  1.100504   1.1703746 ]
     [1.0095401  0.99570805 0.98216826 0.96264714 0.9243948  1.0062507
      0.9584576  1.0262512  0.9985695  1.1360139 ]
     [1.0567318  0.93753624 0.9490582  0.95205057 0.92191374 0.9142274
      1.0194638  1.0218542  1.0609608  1.1662029 ]
     [1.0137744  0.96766555 0.9179035  0.9737802  0.92069834 0.9782479
      1.0531033  0.98723304 1.04638    1.1412137 ]
     [1.0667577  0.93189013 0.9223573  0.936921   0.8979971  0.9597674
      1.0004516  0.96919066 1.0998807  1.2147874 ]
     [0.9775892  0.9686586  0.948501   0.9812417  0.88084155 1.0191782
      1.014613   0.96823657 1.0302212  1.2109194 ]
     [0.9733076  0.938151   0.94007224 0.92271155 0.9100739  0.92999107
      1.036994   0.9904758  1.1387329  1.2194904 ]
     [1.013587   0.9873955  0.97580147 0.9530225  0.89005774 1.0026591
      0.9962303  0.97986627 1.0526042  1.1487764 ]
     [1.0247761  0.94692945 0.9408253  0.939451   0.9161692  0.95160997
      1.0174093  0.98331755 1.0957255  1.183786  ]
     [1.0129678  0.971182   0.9202944  0.9530813  0.91039246 0.9546298
      1.038661   0.999385   1.093362   1.1460437 ]
     [1.0552473  0.9829167  0.9804774  0.9437772  0.87241185 0.93286365
      0.9781116  1.0032139  1.1158873  1.1350925 ]
     [1.0488454  0.96979266 0.9384248  0.9797144  0.9047431  0.96086466
      1.0233924  0.99831223 1.0465243  1.1293858 ]
     [1.0604684  0.9368753  0.9610474  0.9972423  0.89922017 0.95300126
      0.9875939  0.9994408  1.075072   1.130038  ]
     [0.96376413 0.98143935 0.9349103  0.92571354 0.91548157 0.9217865
      1.0219058  0.9930564  1.130607   1.2113357 ]
     [1.0092297  0.9630477  0.93168265 1.0172143  0.94102025 0.97963357
      1.0260687  0.99449515 1.0352484  1.1023599 ]
     [1.0242465  0.9505188  0.94921875 0.9508513  0.8828059  0.9182341
      0.9836159  1.030241   1.1213367  1.1889307 ]
     [1.0200363  0.9521924  0.9458643  0.94377345 0.9299025  0.98155445
      1.0221889  0.9956794  1.0705035  1.1383053 ]
     [0.99562556 0.9682479  0.9430307  0.99415755 0.88225794 0.98142886
      1.0383158  0.9972607  1.056263   1.1434121 ]], shape=(32, 10), dtype=float32)
    FastEstimator-Train: step: 0; loss: 2.29256; 
    FastEstimator-Eval: step: 1; epoch: 0; loss: 2.2865727; min_loss: 2.2865727; since_best_loss: 0; 
    FastEstimator-Finish: step: 1; total_time: 2.5 sec; lenet_lr: 0.001; 

