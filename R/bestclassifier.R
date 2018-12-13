#' @title Identify Best Binary Classification Model
#' @description
#' \code{bestclassifier} trains up to eight binary classification models
#' on a dataset and identifies the most predictive model according to
#' either AUC or accuracy
#' @details
#' This function uses the caret package to employ as many as eight binary 
#' classification models on a dataset, allowing the user to train
#' logistic regression, lasso regression, random forest, extreme gradient 
#' boosting, support vector machine, artificial neural network, latent 
#' dirichlet allocation, and k nearest neighbors models. After training the 
#' models, the function prints a bar graph depicting the most predictive
#' machine learning model based on AUC or Accuracy and outputs the name of
#' the best model on the training dataset as well as its performance results.
#' The function then employs the best model on a testing dataset and returns
#' a confusion matrix describing the model's predictive results.
#' @param data a data frame containing the variables in the model
#' @param form an object of class formula, relating the binary dependent variable to the independent variables
#' @param p the proportion of data used on the training dataset
#' @param method the resampling method employed by the machine learning models
#' @param number either the number of folds or the number of resampling iterations
#' @param repeats the number of complete sets of folds to compute for repeated k-fold cross validation
#' @param tuneLength an integer depicting the number of levels for each tuning parameter to be generated
#' @param positive the factor (written as a character string) that corresponds to a "positive" result in your data
#' @param model the specific binary classification machine learning models to be trained on the data
#' @param set_seed the seed used for the models
#' @param subset_train the percentage of the training data used to train the model
#' @param desired_metric whether the user wants to use AUC or Accuracy to evaluate the models
#' @import ggplot2
#' @import caret
#' @import dplyr
#' @import e1071
#' @export
#' @return a confusion matrix of the best binary classification model
#' @author Shane Ross <saross@@wesleyan.edu>
#' @examples
#' data(Ionosphere, package = "mlbench")
#' Ionosphere <- Ionosphere[-2]
#' names <- c("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "aa", "bb", "cc", "dd", "ee", "ff", "gg", "Class")
#' names(Ionosphere) <- names
#' bestclassifier(data = Ionosphere, form = Class ~ ., method = "repeatedcv",
#'                 number = 5, repeats = 2, tuneLength = 5, positive = "good",
#'                 model = c("log_reg", "lasso", "rf", "svm", "ann"), set_seed = 1234,
#'                 subset_train = 1.0, desired_metric = "ROC")


bestclassifier <- function(data, form, p = .7, method = c("boot", "boot632", "optimism_boot", "boot_all", "cv", "repeatedcv", "LOOCV", "LGOCV", "none", "oob", "adaptive_cv", "adaptive_boot", "adaptive_LGOCV"),
                            number = 10, repeats = ifelse(grepl("[d_]cv$", method), 1, NA),
                            tuneLength = 5, positive, model = c("log_reg", "lasso", "rf",
                                                                "svm", "xgboost", "ann",
                                                                "lda", "knn"),
                            set_seed = 1234, subset_train = 1.0,
                            desired_metric = c("ROC", "Accuracy")) {
  
  predictive_power <- c()
  names <- c()
  
  i = 1
  
  if (!(class(data) == "data.frame")) {
    stop("data must be of class data.frame")
  }
  if (ncol(data) <= 1) {
    stop("data must have more than one column")
  }
  if (nrow(data) <= 1) {
    stop("data must have more than one row")
  }
  
  if (class(form) != "formula") {
    stop("form must be a valid formula")
  }
  
  if (sum(complete.cases(data)) < length(data[[form[[2]]]])) {
    msg <- print("remove rows with missing values (Y/N)? ")
    answer <- readline(msg)
    if (toupper(answer == "Y")) {
      data <- data[complete.cases(data), ]
    }
  }
  
  if (typeof(p) != "double") {
    stop("p must be of type double")
  } else if (p < 0 | p > 1) {
    stop("p must be between 0 and 1")
  }
  
  method <- match.arg(method)
  
  if (!(is.numeric(number))) {
    stop("number of folds must be numeric")
  }
  round(number, digits = 0)
  
  if (!(is.na(repeats))) {
    round(repeats, digits = 0)
  }
  
  if (!(is.numeric(tuneLength))) {
    stop("tuneLength must be numeric")
  }
  round(tuneLength, digits = 0)
  
  if (!(is.character(positive))) {
    stop("positive must be a character")
  }
  
  if (sum(data[[form[[2]]]] == positive, na.rm = TRUE) == 0) {
    stop("positive must be a valid argument of the dependent variable")
  }
  
  if (!is.numeric(set_seed)) {
    stop("seed must be numeric")
  }
  
  desired_metric <- match.arg(desired_metric)
  
  if (!is.numeric(subset_train)) {
    stop("subset must be numeric")
  } else if (subset_train < 0 | subset_train > 1) {
    stop("subset_train must be in between 0 and 1")
  }
  
  set.seed(set_seed)
  index <- createDataPartition(data[[form[[2]]]], p = p, list = FALSE)
  train_data <- data[index, ]
  train_data <- sample_n(train_data, nrow(train_data)*subset_train)
  test_data <- data[-index, ]
  
  tuning_parameters <- trainControl(method = method,
                                    number = number,
                                    repeats = repeats,
                                    summaryFunction = twoClassSummary,
                                    classProbs = TRUE,
                                    verboseIter = FALSE)
  
  
  if ("log_reg" %in% model) {
    
    logistic_model <- train(form = form,
                            data = train_data,
                            method = "glm",
                            family = "binomial",
                            trControl = tuning_parameters,
                            tuneLength = tuneLength)
    
    if (desired_metric == "ROC") {
      a_metric <- max(logistic_model$results[[2]])
    } else if (desired_metric == "Accuracy") {
      a_metric <- (max(logistic_model$results[[3]]) + max(logistic_model$results[[4]]))/2
    }
    
    a_metric <- round(a_metric, 4)
    
    predictive_power <- c(predictive_power, a_metric)
    log_reg_name <- "log_reg"
    names(predictive_power)[[i]] <- log_reg_name
    i = i + 1
  }
  
  if ("lasso" %in% model) {
    
    lasso_model <- train(form = form,
                         data = train_data,
                         method = "glmnet",
                         family = "binomial",
                         trControl = tuning_parameters,
                         tuneLength = 5)
    
    if (desired_metric == "ROC") {
      b_metric <- max(lasso_model[["results"]][[3]])
    } else if (desired_metric == "Accuracy") {
      b_metric <- (max(lasso_model$results[[4]]) + max(lasso_model$results[[5]]))/2
    }
    
    b_metric <- round(b_metric, 4)
    
    predictive_power <- c(predictive_power, b_metric)
    lasso_name <- "lasso_reg"
    names(predictive_power)[[i]] <- lasso_name
    i = i + 1
  }
  
  if ("rf" %in% model) {
    
    rf_model <- train(form = form,
                      data = train_data,
                      method = "rf",
                      trControl = tuning_parameters,
                      tuneLength = tuneLength)
    
    if (desired_metric == "ROC") {
      c_metric <- max(rf_model$results[[2]])
    } else if (desired_metric == "Accuracy") {
      c_metric <- (max(rf_model$results[[3]]) + max(rf_model$results[[4]]))/2
    }
    
    c_metric <- round(c_metric, 4)
    
    predictive_power <- c(predictive_power, c_metric)
    rf_name <- "rf"
    names(predictive_power)[[i]] <- rf_name
    i = i + 1
  }
  
  if ("svm" %in% model) {
    
    svm_model <- train(form = form,
                       data = train_data,
                       method = "svmRadial",
                       trControl = tuning_parameters,
                       tuneLength = tuneLength)
    
    if (desired_metric == "ROC") {
      d_metric <- max(svm_model$results[[3]])
    } else if (desired_metric == "Accuracy") {
      d_metric <- (max(svm_model$results[[4]]) + max(svm_model$results[[5]]))/2
    }
    
    d_metric <- round(d_metric, 4)
    
    predictive_power <- c(predictive_power, d_metric)
    svm_name <- "svm"
    names(predictive_power)[[i]] <- svm_name
    i = i + 1
  }
  
  if ("xgboost" %in% model) {
    
    xgb_model <- train(form = form,
                       data = train_data,
                       method = "xgbTree",
                       trControl = tuning_parameters,
                       tuneLength = tuneLength)
    
    if (desired_metric == "ROC") {
      e_metric <- max(xgb_model$results[[8]])
    } else if (desired_metric == "Accuracy") {
      e_metric <- (max(xgb_model$results[[9]]) + max(xgb_model$results[[10]]))/2
    }
    
    e_metric <- round(e_metric, 4)
    
    predictive_power <- c(predictive_power, e_metric)
    xgboost_name <- "xgboost"
    names(predictive_power)[[i]] <- xgboost_name
    i = i + 1
    
  }
  
  if ("ann" %in% model) {
    ann_model <- train(form = form,
                       data = train_data,
                       method = "nnet",
                       trControl = tuning_parameters,
                       tuneLength = tuneLength,
                       linout = FALSE,
                       trace = FALSE)
    
    if (desired_metric == "ROC") {
      f_metric <- max(ann_model$results[[3]])
    } else if (desired_metric == "Accuracy") {
      f_metric <- (max(ann_model$results[[4]]) + max(ann_model$results[[5]]))/2
    }
    
    f_metric <- round(f_metric, 4)
    
    predictive_power <- c(predictive_power, f_metric)
    ann_name <- "ann"
    names(predictive_power)[[i]] <- ann_name
    i = i + 1
  }
  
  if ("lda" %in% model) {
    
    lda_model <- train(form = form,
                       data = train_data,
                       method = "lda",
                       trControl = tuning_parameters,
                       preProcess = c("center", "scale"),
                       tuneLength = tuneLength)
    
    
    if (desired_metric == "ROC") {
      g_metric <- max(lda_model$results[[2]])
    } else if (desired_metric == "Accuracy") {
      g_metric <- (max(lda_model$results[[3]]) + max(lda_model$results[[4]]))/2
    }
    
    g_metric <- round(g_metric, 4)
    
    predictive_power <- c(predictive_power, g_metric)
    lda_name <- "lda"
    names(predictive_power)[[i]] <- lda_name
    i = i + 1
  }
  
  if ("knn" %in% model) {
    
    knn_model <- train(form = form,
                       data = train_data,
                       method = "knn",
                       trControl = tuning_parameters,
                       preProcess = c("center", "scale"),
                       tuneLength = tuneLength)
    
    if (desired_metric == "ROC") {
      h_metric <- max(knn_model$results[[2]])
    } else if (desired_metric == "Accuracy") {
      h_metric <- (max(knn_model$results[[3]]) + max(knn_model$results[[4]]))/2
    }
    
    h_metric <- round(h_metric, 4)
    
    predictive_power <- c(predictive_power, h_metric)
    knn_name <- "knn"
    names(predictive_power)[[i]] <- knn_name
  }
  
  model_names <- names(predictive_power)
  
  best_model <- model_names[which.max(predictive_power)]
  
  if (desired_metric == "ROC") {
    cat(paste("Your best binary classification model is", best_model,
              "yielding an AUC of", max(predictive_power), "\n", "\n"))
  } else if (desired_metric == "Accuracy") {
    cat(paste("Your best binary classification model is", best_model,
              "yielding an Accuracy of", max(predictive_power), "\n",
              "\n"))
  }
  
  predictive_power <- as.data.frame(predictive_power)
  rownames(predictive_power) <- model_names
  
  predictive_power$is_max <- ifelse(predictive_power$predictive_power == max(predictive_power), TRUE, FALSE)
  
  predictive_power <- predictive_power[order(rownames(predictive_power)), ]
  model_names <- rownames(predictive_power)
  
  print(ggplot(data = predictive_power, aes(x = model_names,
                                            y = predictive_power)) +
          geom_bar(stat = "identity", color = ifelse(predictive_power$is_max == TRUE, "red", "skyblue"), fill = "white") +
          geom_text(aes(label = predictive_power), vjust = 1.6,
                    color = "black", size = 3.5) +
          ylab(ifelse(desired_metric == "ROC", "ROC", "Accuracy")) +
          xlab("Models") +
          ggtitle(ifelse(desired_metric == "ROC", "AUC on training data",
                         "Accuracy on training data")) +
          theme_bw() +
          theme(plot.title = element_text(hjust = 0.5)))
  
  
  if (best_model == "log_reg") {
    
    set.seed(set_seed)
    index <- createDataPartition(data[[form[[2]]]], p = p, list = FALSE)
    test_data <- data[-index, ]
    
    pred <- predict(logistic_model,  newdata = test_data)
    return(confusionMatrix(pred, test_data[[form[[2]]]], positive = positive))
  } else if (best_model == "lasso_reg") {
    
    set.seed(set_seed)
    index <- createDataPartition(data[[form[[2]]]], p = p, list = FALSE)
    test_data <- data[-index, ]
    
    pred <- predict(lasso_model,  newdata = test_data)
    return(confusionMatrix(pred, test_data[[form[[2]]]], positive = positive))
  } else if (best_model == "rf") {
    
    set.seed(set_seed)
    index <- createDataPartition(data[[form[[2]]]], p = p, list = FALSE)
    test_data <- data[-index, ]
    
    pred <- predict(rf_model,  newdata = test_data)
    return(confusionMatrix(pred, test_data[[form[[2]]]], positive = positive))
  } else if (best_model == "svm") {
    
    set.seed(set_seed)
    index <- createDataPartition(data[[form[[2]]]], p = p, list = FALSE)
    test_data <- data[-index, ]
    
    pred <- predict(svm_model,  newdata = test_data)
    return(confusionMatrix(pred, test_data[[form[[2]]]], positive = positive))
  } else if (best_model == "xgboost") {
    
    set.seed(set_seed)
    index <- createDataPartition(data[[form[[2]]]], p = p, list = FALSE)
    test_data <- data[-index, ]
    
    pred <- predict(xgb_model,  newdata = test_data)
    return(confusionMatrix(pred, test_data[[form[[2]]]], positive = positive))
  } else if (best_model == "ann") {
    
    set.seed(set_seed)
    index <- createDataPartition(data[[form[[2]]]], p = p, list = FALSE)
    test_data <- data[-index, ]
    
    pred <- predict(ann_model,  newdata = test_data)
    return(confusionMatrix(pred, test_data[[form[[2]]]], positive = positive))
  } else if (best_model == "lda") {
    
    set.seed(set_seed)
    index <- createDataPartition(data[[form[[2]]]], p = p, list = FALSE)
    test_data <- data[-index, ]
    
    pred <- predict(lda_model,  newdata = test_data)
    return(confusionMatrix(pred, test_data[[form[[2]]]], positive = positive))
  } else if (best_model == "knn") {
    
    set.seed(set_seed)
    index <- createDataPartition(data[[form[[2]]]], p = p, list = FALSE)
    test_data <- data[-index, ]
    
    pred <- predict(knn_model,  newdata = test_data)
    return(confusionMatrix(pred, test_data[[form[[2]]]], positive = positive))
  }
  
}
