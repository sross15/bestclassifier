# Identify Best Binary Classification Model

## Overview: bestclassifier

The bestclassifier function was created in order to simplify the process of identifying the best binary classification model for a given dataset. While this process generally involves an arduous process of finding all of the relevant binary classification models, tuning them, training them, and then comparing them to find the best result, the bestclassifier function combines all of these steps into one user-friendly function.

## Installation

To download this package use the following code:

``` r
if (!require(devtools)) {
install.packages("devtools") }
devtools::install_github("sross15/bestclassifier", build_vignettes = TRUE)
```

The source code is available [here](https://github.com/sross15/bestclassifier).

## Usage

The bestclassifier function trains as many as eight machine learning binary classification models in order to identify the best predictive model for a given dataset. The available models include:

  - logistic regression

  - lasso regression

  - random forest

  - extreme gradient boosting
  
  - support vector machine
  
  - artificial neural network
  
  - latent dirichlet allocation
  
  - k nearest neighbors
  

Once identifying the best machine learning model, the bestclassifier function will: 

  - print a bar graph depicting the performance of each model on the training data

  - print the name of the best binary classification model along with its predictive performance score (either AUC or Accuracy depending upon what the user selects)

  - employ the best trained model on an unseen testing data and return a confusion matrix with overall performance results

## Example 

Below is an example of the bestclassifier function used on the CCD dataset:

``` r
library(bestclassifier)

bestclassifier(data = CCD, form = default.payment.next.month ~ ., p = 0.7, 
              method = "repeatedcv", number = 5, repeats = 1, tuneLength = 5, 
              positive ="Default", model = c("log_reg", "lasso", "rf", "svm", "lda", 
              "knn", "ann", "xgboost"), set_seed = 1234, subset_train = .01, 
              desired_metric = "ROC")
```

## Getting Help

If you find a clear bug, please file a minimal reproducible example on github. For questions and other comments please use community.rstudio.com.