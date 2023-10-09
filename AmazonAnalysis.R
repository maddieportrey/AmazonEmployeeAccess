library(tidyverse)
library(vroom)
library(tidymodels)
library(ggmosaic)
library(embed)
setwd("./AmazonEmployeeAccess")
#112 columns

amazonTrain <- vroom("./train.csv")
amazonTest <- vroom("./test.csv")
amazonTrain$ACTION <- as.factor(amazonTrain$ACTION)

my_recipe <- recipe(ACTION~., data = amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_factor_predictors(), threshold = .01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor)
  
#prep <- prep(my_recipe)
#baked <- bake(prep, new_data = amazonTrain)

my_mod <- logistic_reg() %>%
  set_engine("glm")

amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = amazonTrain)

amazon_preds <- predict(amazon_workflow, 
                        new_data = amazonTest,
                        type="prob")

preds <- cbind(amazonTest$id, amazon_preds$.pred_1)
colnames(preds) <- c("Id","ACTION")
preds <- as.data.frame(preds)
vroom_write(preds, "amazon_predictions.csv", ",")
