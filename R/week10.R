# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven) 
library(janitor) # for remove_empty
library(caret)
set.seed(12138)

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
                          search = "grid",
                          summaryFunction = defaultSummary, ## For summary
                          verboseIter = TRUE) ## Printing training log

## OLS Regression
model.ols <- train(`work hours` ~ ., 
                   gss_train_tbl,
                   method = "lm",  
                   metric = "Rsquared",
                   preProcess = "medianImpute", ## Impute Median
                   na.action = na.pass, ## So it will impute
                   trControl = myControl)
## Warning: In predict.lm(modelFit, newdata) :
## prediction from rank-deficient fit; attr(*, "non-estim") has doubtful cases






