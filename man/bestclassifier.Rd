% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/bestclassifier.R
\name{bestclassifier}
\alias{bestclassifier}
\title{Identify Best Binary Classification Model}
\usage{
bestclassifier(data, form, p = 0.7, method = c("boot", "boot632",
  "optimism_boot", "boot_all", "cv", "repeatedcv", "LOOCV", "LGOCV",
  "none", "oob", "adaptive_cv", "adaptive_boot", "adaptive_LGOCV"),
  number = 10, repeats = ifelse(grepl("[d_]cv$", method), 1, NA),
  tuneLength = 5, positive, model = c("log_reg", "lasso", "rf", "svm",
  "xgboost", "ann", "lda", "knn"), set_seed = 1234, subset_train = 1,
  desired_metric = c("ROC", "Accuracy"))
}
\arguments{
\item{data}{a data frame containing the variables in the model}

\item{form}{an object of class formula, relating the binary dependent variable to the independent variables}

\item{p}{the proportion of data used on the training dataset}

\item{method}{the resampling method employed by the machine learning models}

\item{number}{either the number of folds or the number of resampling iterations}

\item{repeats}{the number of complete sets of folds to compute for repeated k-fold cross validation}

\item{tuneLength}{an integer depicting the number of levels for each tuning parameter to be generated}

\item{positive}{the factor (written as a character string) that corresponds to a "positive" result in your data}

\item{model}{the specific binary classification machine learning models to be trained on the data}

\item{set_seed}{the seed used for the models}

\item{subset_train}{optional parameter used to reduce the size of the training dataset in order to speed up binary classification model creation. This parameter is a numeric object between 0 and 1.}

\item{desired_metric}{whether the user wants to use AUC or Accuracy to evaluate the models}
}
\value{
the best binary classification model
}
\description{
\code{bestclassifier} trains up to eight binary classification models
on a dataset and identifies the most predictive model according to
either AUC or Accuracy
}
\details{
This function uses the caret package to train as many as eight binary 
classification models on a dataset, allowing the user to build
logistic regression, lasso regression, random forest, extreme gradient 
boosting, support vector machine, artificial neural network, latent 
dirichlet allocation, and k nearest neighbors models. After training the 
models, the function prints a bar graph depicting the most predictive
machine learning model based on AUC or Accuracy and outputs the name of
the best model on the training dataset as well as its predictive performance.
The function then implements the best model on a testing dataset and prints
a confusion matrix with the model's predictive performance. The function returns
the best model.
}
\examples{
data(Ionosphere, package = "mlbench")
Ionosphere <- Ionosphere[-2]
names <- c("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "aa", "bb", "cc", "dd", "ee", "ff", "gg", "Class")
names(Ionosphere) <- names
bestclassifier(data = Ionosphere, form = Class ~ ., method = "repeatedcv",
                number = 5, repeats = 2, tuneLength = 5, positive = "good",
                model = c("log_reg", "lasso", "rf"), set_seed = 1234,
                subset_train = 1.0, desired_metric = "ROC")
}
\author{
Shane Ross <saross@wesleyan.edu>
}
