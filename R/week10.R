# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven) 
library(janitor) # for remove_empty
library(caret)
set.seed(12138)
## Loaded Random Forest and xgbboost package when running caret

# Data Import and Cleaning
## Read in data
GSS2016 <- read_sav("../data/GSS2016.sav") 
gss_tbl <- GSS2016 %>%
  filter(!is.na(MOSTHRS)) %>% ## Remove missing value in MOSTHRS
  rename(`work hours` = MOSTHRS) %>% ## Rename MOSTHERS
  select(-HRS1, -HRS2) %>% ## Remove variables
  remove_empty("cols", cutoff = 0.25) %>%  ## Remove rows with more than 25% missingness
  sapply(as.numeric) %>% ## Important! Otherwise will wrong the error:Wrong model type for classification
  as_tibble()

# Visualization
gss_tbl %>%
  ggplot(aes(x = `work hours`)) +
  geom_histogram() # Draw histogram

# Analysis

## Split Datasets
### Randomly sort row numbers
random_sample <- sample(nrow(gss_tbl))
### Shuffle dataset
gss_shuffle_tbl <- gss_tbl[random_sample, ]
### Find the 75% index
index <- round(nrow(gss_tbl) * 0.75)
### Create train data
gss_train_tbl <- gss_shuffle_tbl[1:index, ]
### Create test data
gss_test_tbl <- gss_shuffle_tbl[(index+1):nrow(gss_tbl), ]
### Create index for 10 foldes
fold_indices <- createFolds(gss_train_tbl$`work hours`, 10) ## Important to specify to `work hours` here, otherwise model won't run
## Set up train control
myControl <- trainControl(method = "cv", # Cross-Validation
                          indexOut = fold_indices, 
                          number = 10,  #10 folds
                          verboseIter = TRUE) ## Printing training log

## OLS Regression
### No hyperparmeter here cuz the model.ols$bestTune shows intercept = TRUE, which is default
model_ols <- train(`work hours` ~ ., 
                   gss_train_tbl,
                   method = "lm",  
                   metric = "Rsquared",
                   preProcess = "medianImpute", ## Impute Median
                   na.action = na.pass, ## So it will impute
                   trControl = myControl)
## Warning: In predict.lm(modelFit, newdata) :
## prediction from rank-deficient fit; attr(*, "non-estim") has doubtful cases
## saveRDS(model_ols, "OLSmodel.RDS")
ols_predict <- predict(model_ols, gss_test_tbl, na.action = na.pass)


## elasticnet model
### Specify a grid
elastic_grid <- expand.grid(alpha = seq(0, 1, by = 0.1),
                            lambda = seq(0, 2, by = 0.1))
# Build the model
model_elastic <- train(`work hours` ~ ., 
                       data = gss_train_tbl,
                       method = "glmnet",  
                       preProcess = "medianImpute", ## Impute Median
                       na.action = na.pass, ## So it will impute
                       trControl = myControl,
                       tuneGrid = elastic_grid)
### Aggregating results
### Selecting tuning parameters
### Fitting alpha = 1, lambda = 0.1 on full training set
## saveRDS(model_elastic, "Elasticmodel.RDS")
elastic_predict <- predict(model_elastic, gss_test_tbl, na.action = na.pass)


## Random Forest Model
### Specify a grid
rf_grid <- expand.grid(mtry = seq(10, 500, by = 100),
                       splitrule = c("variance", "extratrees"), # Beta is not for regression with 0 and 1 only 
                       min.node.size = 5) # Apparently this is the number for regression
### Build the model
model_rf <- train(`work hours` ~ ., 
                  data = gss_train_tbl,
                  method = "ranger",  
                  preProcess = "medianImpute", ## Impute Median
                  na.action = na.pass, ## So it will impute
                  trControl = myControl,
                  tuneGrid = rf_grid)
### Selecting tuning parameters
### Fitting mtry = 210, splitrule = variance, min.node.size = 5 on full training set
## saveRDS(model_rf, "RandomForest.RDS")
rf_predict <- predict(model_rf, gss_test_tbl, na.action = na.pass)

## eXtreme Gradient Boosting Model
### Specify a grid
xgb_grid <- expand.grid(nrounds = seq(5, 50, by = 10),
                        lambda = c(0, 0.1, 2),
                        alpha = seq(0, 1, by = 0.1),
                        eta = c(0.5, 1))
### Build Model
model_xgb <- train(`work hours` ~ ., 
                   data = gss_train_tbl,
                   method = "xgbLinear",  
                   preProcess = "medianImpute", ## Impute Median
                   na.action = na.pass, ## So it will impute
                   trControl = myControl,
                   tuneGrid = xgb_grid)

### saveRDS(model_xgb, "Xgb.RDS")
xgb_predict <- predict(model_xgb, gss_test_tbl, na.action = na.pass)

# Publication
### Calculate R^2 first
R2_ols <- cor(ols_predict, gss_test_tbl$`work hours`)^2
R2_elastic <- cor(elastic_predict, gss_test_tbl$`work hours`)^2
R2_rf <- cor(rf_predict, gss_test_tbl$`work hours`) ^2
R2_xgb <- cor(xgb_predict, gss_test_tbl$`work hours`) ^2

### Build tibble of results and format
table1_tbl <- tibble(
  algo = c("OLS regression", "Elastic Net", "Random Forest", "XGB"),
  cv_rsq = c(sub("^0\\.", ".", formatC(model_ols$results$Rsquared, format = 'f', digits = 2)),
             sub("^0\\.", ".", formatC(max(model_elastic$results$Rsquared), format = 'f', digits = 2)),
             sub("^0\\.", ".", formatC(max(model_rf$results$Rsquared), format = 'f', digits = 2)),
             sub("^0\\.", ".", formatC(max(model_xgb$results$Rsquared), format = 'f', digits = 2))),
  ho_rsq = c(sub("^0\\.", ".", formatC(R2_ols, format = 'f', digits = 2)),
             sub("^0\\.", ".", formatC(R2_elastic, format = 'f', digits = 2)),
             sub("^0\\.", ".", formatC(R2_rf, format = 'f', digits = 2)),
             sub("^0\\.", ".", formatC(R2_xgb, format = 'f', digits = 2)))
)

### Answer Questions

# 1. The results for OLS model is very bad (R^2 = 0.07), but for other three models are quite similar.
#   I think this happen because in the regrression model we have more predictors than observations, 
#   which makes it very hard for OLS to generate good results.

# 2. The k-folds have better statistics than holds data. I think it is because we are fitting model
#.   to new data which may results in worse fitting. 

# 3. I will choose random forest prediction. I have two concerns: time-wise and predict-ability wise.
#. Time wisely, XGB model takes a long time (could also be due to my hyperparameters) and is therefore 
#. in-efficient, especially if I'm going to fit a larger sample. It might perform better when there are 
#. less prredictors. Although elastic net takes the shortest time to perform, it's predictive ability
#. is considerably worse than random forest and XGB. Therefore, random forest takes the medium amount of
#. time and returns acceptable results.
#  I think OLS may also perform fine if we have less predictors and a larger sample size. But based on this
#  particular situation, it performs pretty bad. 