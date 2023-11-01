# library(tidyverse)
# library(vroom)
# library(tidymodels)
# library(embed)
# library(ranger)
# library(doParallel)
# 
# #112 columns
# 
# amazonTrain <- vroom("./train.csv")
# amazonTest <- vroom("./test.csv")
# amazonTrain$ACTION <- as.factor(amazonTrain$ACTION)
# 
# my_recipe <- recipe(ACTION~., data = amazonTrain) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   step_other(all_nominal_predictors(), threshold = .001) %>%
#   #step_dummy(all_nominal_predictors()) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #%>%
#   #step_mutate_at(all_numeric_predictors(), fn = factor)
# 
# # my_mod <- logistic_reg() %>%
# #   set_engine("glm")
# #
# # amazon_workflow <- workflow() %>%
# #   add_recipe(my_recipe) %>%
# #   add_model(my_mod) %>%
# #   fit(data = amazonTrain)
# #
# # amazon_preds <- predict(amazon_workflow,
# #                         new_data = amazonTest,
# #                         type="prob")
# #
# # preds <- cbind(amazonTest$id, amazon_preds$.pred_1)
# # colnames(preds) <- c("Id","ACTION")
# # preds <- as.data.frame(preds)
# # vroom_write(preds, "amazon_predictions.csv", ",")
# 
# # my_mod <- logistic_reg(mixture = tune(),
# #                        penalty = tune()) %>%
# #   set_engine("glmnet")
# 
# my_mod <- rand_forest(mtry = tune(),
#                       min_n = tune(),
#                       trees = 500) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# amazon_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(my_mod)
# 
# # tuning_grid <- grid_regular(penalty(),
# #                             mixture(),
# #                             levels = 5)
# 
# tuning_grid <- grid_regular(mtry(range = c(1,9)),
#                             min_n(),
#                             levels = 5)
# 
# folds <- vfold_cv(amazonTrain, v = 10, repeats = 1)
# cl <- makePSOCKcluster(10)
# registerDoParallel(cl)
# CV_results <- amazon_workflow %>%
#   tune_grid(resamples = folds,
#             grid = tuning_grid,
#             metrics = metric_set(roc_auc))
# stopCluster(cl)
# bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# final_wf <- amazon_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data = amazonTrain)
# bestTune
# amazon_preds <- final_wf %>% predict(new_data = amazonTest, type = "prob")
# 
# preds <- cbind(amazonTest$id, amazon_preds$.pred_1)
# colnames(preds) <- c("Id","ACTION")
# preds <- as.data.frame(preds)
# vroom_write(preds, "amazon_predictions.csv", ",")

#random forest on smote balanced data


library(tidyverse)
library(vroom)
library(tidymodels)
library(embed)
library(ranger)
library(doParallel)
library(themis)

amazonTrain <- vroom("./train.csv")
amazonTest <- vroom("./test.csv")
amazonTrain$ACTION <- as.factor(amazonTrain$ACTION)

my_recipe <- recipe(ACTION~., data = amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=.9) %>%
  step_smote(all_outcomes(), neighbors = 5) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data = amazonTrain)

my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

tuning_grid <- grid_regular(mtry(range = c(1,(ncol(baked)-1))),
                            min_n(),
                            levels = 3)

folds <- vfold_cv(amazonTrain, v = 3, repeats = 1)
cl <- makePSOCKcluster(10)
registerDoParallel(cl)
CV_results <- amazon_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))
stopCluster(cl)
bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <- amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazonTrain)

amazon_preds <- final_wf %>% predict(new_data = amazonTest, type = "prob")

preds <- cbind(amazonTest$id, amazon_preds$.pred_1)
colnames(preds) <- c("Id","ACTION")
preds <- as.data.frame(preds)
vroom_write(preds, "amazon_predictions.csv", ",")
