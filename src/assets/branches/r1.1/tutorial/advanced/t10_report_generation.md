# Advanced Tutorial 10: Automated Report Generation

## Overview
In this tutorial, we will discuss:
* [Overview and Dependencies](./tutorials/r1.1/advanced/t10_report_generation#ta10od)
* [Traceability](./tutorials/r1.1/advanced/t10_report_generation#ta10t)
* [Test Report](./tutorials/r1.1/advanced/t10_report_generation#ta10tr)

## Preliminary Setup

Let's get some imports and object construction out of the way:


```python
import tempfile
import os
import numpy as np
import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy

root_output_dir = tempfile.mkdtemp()

def get_estimator(extra_traces):
    # step 1
    train_data, eval_data = mnist.load_data()
    test_data = eval_data.split(100)
    test_data['id'] = [i for i in range(len(test_data))]  # Assign some data ids for the test report to look at
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           test_data=test_data,
                           batch_size=32,
                           ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])

    # step 2
    model = fe.build(model_fn=LeNet, optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=root_output_dir, metric="accuracy", save_best_mode="max"),
        LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=3750, init_lr=1e-3))
    ]
    traces.extend(extra_traces)
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=2,
                             traces=traces,
                             max_train_steps_per_epoch=100,
                             max_eval_steps_per_epoch=100,
                             log_steps=10)
    return estimator
```

<a id='ta10od'></a>

## Overview and Dependencies
FastEstimator provides Traces which allow you to automatically generate traceability documents and test reports. These reports are written in the LaTeX file format, and then automatically compiled into PDF documents if you have LaTeX installed on your machine. If you don't have LaTeX installed on your training machine, you can still generate the report files and then move them to a different computer in order to compile them manually. Generating traceability documents also requires GraphViz which, unlike LaTeX, must be installed in order for training to proceed. 

```
Installing Dependencies:
    On Linux: 
        apt-get install -y graphviz texlive-latex-base texlive-latex-extra
    On SageMaker:
        unset PYTHONPATH
        export DEBIAN_FRONTEND=noninteractive
        apt-get install -y graphviz texlive-latex-base texlive-latex-extra
    On Mac:
        brew install graphviz
        brew cask install mactex
    On Windows:
        winget install graphviz
        winget install TeXLive
```

<a id='ta10t'></a>

## Traceability
Traceability reports are designed to capture all the information about the state of your system when an experiment was run. The report will include training graphs, operator architecture diagrams, model architecture diagrams, a summary of your system configuration, and the values of all variables used to instantiate objects during training. It will also automatically save a copy of your log output to disk, which can be especially useful for comparing different experiment configurations without worrying about forgetting what settings were used for each run. To generate this report, simply add a Traceability trace to your list of traces:


```python
from fastestimator.trace.io import Traceability

save_dir = os.path.join(root_output_dir, 'report')
est = get_estimator([Traceability(save_dir)])

print(f"The root save directory is: {root_output_dir}")
print(f"The traceability report will be written to: {save_dir}")
print(f"Logs and images from the report will be written to: {os.path.join(save_dir, 'resources')}")
```

    The root save directory is: /tmp/tmpwps4y0dd
    The traceability report will be written to: /tmp/tmpwps4y0dd/report
    Logs and images from the report will be written to: /tmp/tmpwps4y0dd/report/resources


When using Traceability, you must pass a summary name to the Estimator.fit() call. This will become the name of your report.


```python
est.fit("Sample MNIST Report")
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; num_device: 1; logging_interval: 10; 
    FastEstimator-Train: step: 1; ce: 2.3301315; model_lr: 0.0009999998; 
    FastEstimator-Train: step: 10; ce: 2.0306354; steps/sec: 447.07; model_lr: 0.0009999825; 
    FastEstimator-Train: step: 20; ce: 1.5300003; steps/sec: 528.49; model_lr: 0.0009999298; 
    FastEstimator-Train: step: 30; ce: 0.8917824; steps/sec: 524.95; model_lr: 0.0009998423; 
    FastEstimator-Train: step: 40; ce: 0.56758463; steps/sec: 563.14; model_lr: 0.0009997196; 
    FastEstimator-Train: step: 50; ce: 0.33073947; steps/sec: 560.82; model_lr: 0.0009995619; 
    FastEstimator-Train: step: 60; ce: 0.5340511; steps/sec: 553.62; model_lr: 0.0009993691; 
    FastEstimator-Train: step: 70; ce: 0.4809867; steps/sec: 569.04; model_lr: 0.0009991414; 
    FastEstimator-Train: step: 80; ce: 0.17024752; steps/sec: 688.01; model_lr: 0.0009988786; 
    FastEstimator-Train: step: 90; ce: 0.29996952; steps/sec: 659.72; model_lr: 0.0009985808; 
    FastEstimator-Train: step: 100; ce: 0.2563717; steps/sec: 671.72; model_lr: 0.0009982482; 
    FastEstimator-Train: step: 100; epoch: 1; epoch_time: 0.76 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpwps4y0dd/model_best_accuracy.h5
    FastEstimator-Eval: step: 100; epoch: 1; ce: 0.3122454; accuracy: 0.9159375; since_best_accuracy: 0; max_accuracy: 0.9159375; 
    FastEstimator-Train: step: 110; ce: 0.32248974; steps/sec: 71.81; model_lr: 0.0009978806; 
    FastEstimator-Train: step: 120; ce: 0.13474032; steps/sec: 489.5; model_lr: 0.000997478; 
    FastEstimator-Train: step: 130; ce: 0.21002704; steps/sec: 581.26; model_lr: 0.0009970407; 
    FastEstimator-Train: step: 140; ce: 0.11852975; steps/sec: 564.03; model_lr: 0.0009965684; 
    FastEstimator-Train: step: 150; ce: 0.36558276; steps/sec: 564.45; model_lr: 0.0009960613; 
    FastEstimator-Train: step: 160; ce: 0.27304798; steps/sec: 556.14; model_lr: 0.0009955195; 
    FastEstimator-Train: step: 170; ce: 0.16349566; steps/sec: 547.83; model_lr: 0.0009949428; 
    FastEstimator-Train: step: 180; ce: 0.2748593; steps/sec: 605.25; model_lr: 0.0009943316; 
    FastEstimator-Train: step: 190; ce: 0.33497655; steps/sec: 679.45; model_lr: 0.0009936856; 
    FastEstimator-Train: step: 200; ce: 0.24099544; steps/sec: 678.38; model_lr: 0.000993005; 
    FastEstimator-Train: step: 200; epoch: 2; epoch_time: 0.27 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpwps4y0dd/model_best_accuracy.h5
    FastEstimator-Eval: step: 200; epoch: 2; ce: 0.20991856; accuracy: 0.92875; since_best_accuracy: 0; max_accuracy: 0.92875; 
    FastEstimator-Finish: step: 200; total_time: 3.82 sec; model_lr: 0.000993005; 
    FastEstimator-Traceability: Report written to /tmp/tmpwps4y0dd/report/sample_mnist_report.pdf





    <fastestimator.summary.summary.Summary at 0x7f1a7cb84080>




    
![png](assets/branches/r1.1/tutorial/advanced/t10_report_generation_files/t10_report_generation_9_2.png)
    


If everything went according to plan, then inside your root save directory you should now have the following files:

```
/report
    sample_mnist_report.pdf
    sample_mnist_report.tex
    /resources
        sample_mnist_report_logs.png
        sample_mnist_report_model.pdf
        sample_mnist_report.txt
```

You could then switch up your experiment parameters and call .fit() with a new experiment name in order to write more reports into the same folder. A call to `fastestimator logs ./resources` would then allow you to easily compare these experiments, as described in [Advanced Tutorial 6](./tutorials/r1.1/advanced/t06_summary)

Our report should look something like this (use Chrome or Firefox to view):


```python
from IPython.display import IFrame
IFrame('../resources/t10a_traceability.pdf', width=600, height=800)
```





<iframe
    width="600"
    height="800"
    src="assets/branches/r1.1/tutorial/../resources/t10a_traceability.pdf"
    frameborder="0"
    allowfullscreen
></iframe>




<a id='ta10tr'></a>

## Test Report
Test Reports can provide an automatically generated overview summary of how well your model is performing. This could be useful if, for example, you needed to submit documentation to a regulatory agency. Test Reports can also be used to highlight particular failure cases so that you can investigate problematic data points in more detail.

The `TestReport` trace takes a list of `TestCase` objects as input. These are further subdivided into two types: aggregate and per-instance. Aggregate test cases run at the end of the test epoch and deal with aggregated information (typically metrics such as accuracy). Per-instance tests run at the end of every step during testing, and are meant to evaluate every element within a batch independently. If your data dictionary happens to contain data instance ids, you can also use these to find problematic inputs. 


```python
from fastestimator.trace.io.test_report import TestCase, TestReport

save_dir = os.path.join(root_output_dir, 'report2')

# Note that the name of the input to the 'criteria' function must match a key in the data dictionary
agg_test_easy = TestCase(description='Accuracy should be greater than 1%', criteria=lambda accuracy: accuracy > 0.01)
agg_test_hard = TestCase(description='Accuracy should be greater than 99%', criteria=lambda accuracy: accuracy > 0.99)

inst_test_hard = TestCase(description='All Data should be correctly classified', criteria=lambda y, y_pred: np.equal(y,np.argmax(y_pred, axis=-1)), aggregate=False, fail_threshold=0)
inst_test_easy = TestCase(description='At least one image should be correctly classified', criteria=lambda y, y_pred: np.equal(y,np.argmax(y_pred, axis=-1)), aggregate=False, fail_threshold=len(est.pipeline.data['test'])-1)

report = TestReport(test_cases=[agg_test_easy, agg_test_hard, inst_test_easy, inst_test_hard], save_path=save_dir, data_id='id')

est = get_estimator([report])

print(f"The root save directory is: {root_output_dir}")
print(f"The test report will be written to: {save_dir}")
print(f"A json summary of the report will be written to: {os.path.join(save_dir, 'resources')}")
```

    The root save directory is: /tmp/tmpwps4y0dd
    The test report will be written to: /tmp/tmpwps4y0dd/report2
    A json summary of the report will be written to: /tmp/tmpwps4y0dd/report2/resources



```python
est.fit("MNIST")
est.test()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 1; num_device: 1; logging_interval: 10; 
    FastEstimator-Train: step: 1; ce: 2.3229642; model1_lr: 0.0009999998; 
    FastEstimator-Train: step: 10; ce: 2.0991652; steps/sec: 587.99; model1_lr: 0.0009999825; 
    FastEstimator-Train: step: 20; ce: 1.6330333; steps/sec: 605.43; model1_lr: 0.0009999298; 
    FastEstimator-Train: step: 30; ce: 1.0917461; steps/sec: 660.18; model1_lr: 0.0009998423; 
    FastEstimator-Train: step: 40; ce: 1.0353419; steps/sec: 737.78; model1_lr: 0.0009997196; 
    FastEstimator-Train: step: 50; ce: 0.5517947; steps/sec: 680.52; model1_lr: 0.0009995619; 
    FastEstimator-Train: step: 60; ce: 0.34303236; steps/sec: 723.04; model1_lr: 0.0009993691; 
    FastEstimator-Train: step: 70; ce: 0.4734753; steps/sec: 626.83; model1_lr: 0.0009991414; 
    FastEstimator-Train: step: 80; ce: 0.35647863; steps/sec: 705.1; model1_lr: 0.0009988786; 
    FastEstimator-Train: step: 90; ce: 0.57413185; steps/sec: 740.04; model1_lr: 0.0009985808; 
    FastEstimator-Train: step: 100; ce: 0.33393666; steps/sec: 697.83; model1_lr: 0.0009982482; 
    FastEstimator-Train: step: 100; epoch: 1; epoch_time: 0.35 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpwps4y0dd/model1_best_accuracy.h5
    FastEstimator-Eval: step: 100; epoch: 1; ce: 0.3576527; accuracy: 0.8921875; since_best_accuracy: 0; max_accuracy: 0.8921875; 
    FastEstimator-Train: step: 110; ce: 0.43586737; steps/sec: 81.36; model1_lr: 0.0009978806; 
    FastEstimator-Train: step: 120; ce: 0.2825139; steps/sec: 618.99; model1_lr: 0.000997478; 
    FastEstimator-Train: step: 130; ce: 0.23715392; steps/sec: 652.88; model1_lr: 0.0009970407; 
    FastEstimator-Train: step: 140; ce: 0.4731403; steps/sec: 701.43; model1_lr: 0.0009965684; 
    FastEstimator-Train: step: 150; ce: 0.12009444; steps/sec: 662.69; model1_lr: 0.0009960613; 
    FastEstimator-Train: step: 160; ce: 0.1876252; steps/sec: 661.42; model1_lr: 0.0009955195; 
    FastEstimator-Train: step: 170; ce: 0.060715243; steps/sec: 679.76; model1_lr: 0.0009949428; 
    FastEstimator-Train: step: 180; ce: 0.22843787; steps/sec: 690.74; model1_lr: 0.0009943316; 
    FastEstimator-Train: step: 190; ce: 0.28863788; steps/sec: 724.46; model1_lr: 0.0009936856; 
    FastEstimator-Train: step: 200; ce: 0.1017374; steps/sec: 695.11; model1_lr: 0.000993005; 
    FastEstimator-Train: step: 200; epoch: 2; epoch_time: 0.26 sec; 
    FastEstimator-BestModelSaver: Saved model to /tmp/tmpwps4y0dd/model1_best_accuracy.h5
    FastEstimator-Eval: step: 200; epoch: 2; ce: 0.25012556; accuracy: 0.9221875; since_best_accuracy: 0; max_accuracy: 0.9221875; 
    FastEstimator-Finish: step: 200; total_time: 3.57 sec; model1_lr: 0.000993005; 
    FastEstimator-Test: step: 200; epoch: 2; accuracy: 0.93; 
    FastEstimator-TestReport: Report written to /tmp/tmpwps4y0dd/report2/mnist_TestReport.pdf





    <fastestimator.summary.summary.Summary at 0x7f1a7cb7f6d8>



If everything went according to plan, then inside your root save directory you should now have the following files:

```
/report2
    mnist_TestReport.pdf
    mnist_TestReport.tex
    /resources
        mnist_TestReport.json
```

Our report should look something like this (use Chrome or Firefox to view):


```python
from IPython.display import IFrame
IFrame('../resources/t10a_test.pdf', width=600, height=800)
```





<iframe
    width="600"
    height="800"
    src="assets/branches/r1.1/tutorial/../resources/t10a_test.pdf"
    frameborder="0"
    allowfullscreen
></iframe>



