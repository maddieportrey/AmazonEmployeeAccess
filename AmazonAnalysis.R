library(tidyverse)
library(vroom)
library(tidymodels)
library(ggmosaic)
library(embed)
setwd("./AmazonEmployeeAccess")
#112 columns

amazonTrain <- vroom("./train.csv")
amazonTest <- vroom("./test.csv")

# ggplot(data = amazonTrain) +
#   geom_mosaic(aes(x = product(ROLE_FAMILY), fill=ACTION))

my_recipe <- recipe(ACTION~., data = amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_factor_predictors(), threshold = .01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazonTrain)
