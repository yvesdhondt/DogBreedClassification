# Image Classification using AWS SageMaker

*Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.*

This project classifies 133 dog breeds. This is achieved by finetuning a resnet18 model and optimizing its hyperparameters.

## Project Set Up and Installation
*Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. *

## Dataset
*The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.*

The dataset used is the dogbreed classification dataset. This dataset contains images on 133 different dog breeds.

### Access
*Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. *

## Hyperparameter Tuning
*What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search*

For this experiment three hyperparameters were tuned:

```python
hyperparameter_ranges = {
    "lr": ContinuousParameter(0.001, 0.1),
    "batch-size": CategoricalParameter([32, 64, 128, 256, 512]),
    "epochs": IntegerParameter(2, 5)
}
```

- Different learning rates allow to find the best trade-off between being unable to learn (too small) and being too unstable (too big)
- Different batch sizes allow to find the model that makes the best trade-off between the increased speed from large batches and backpropagating on small batches
- Different epochs allow to find the model that has the best trade-off between training a lot of cycles (many epochs) and the time savings from having few epochs

Below are some screenshots of the hyperparameter tuning:

![](hpo_s1.PNG)

Here are also screenshots of 2 of the training jobs from this tuning.

JOB 1 (the best model):

![](hpo_t1_0.PNG)

![](hpo_t1_1.PNG)

JOB 2:

![](hpo_t2_0.PNG)

![](hpo_t2_1.PNG)

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
