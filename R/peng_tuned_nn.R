library(tidyverse)
library(tidymodels)
library(brulee)
library(tidypredict)
library(yaml)

tidymodels_prefer()
theme_set(theme_minimal(base_size = 12))
doParallel::registerDoParallel()

penguins <- penguins %>%
  filter(!is.na(bill_length_mm),
         !is.na(bill_depth_mm),
         !is.na(flipper_length_mm),
         !is.na(body_mass_g)) %>%
  select(-c(species,sex)) 

peng_split <- initial_split(penguins, strata = island)
peng_train <- training(peng_split)
peng_test <- testing(peng_split)

peng_folds <- vfold_cv(peng_train, strata = island, v = 5)

peng_recipe <- recipe(island ~ ., data = peng_train) %>%
  step_normalize(all_numeric_predictors()) 

# A Tuned MLP

nn_spec <- mlp(hidden_units = 10,
               epochs = 150L,
               learn_rate = tune()) %>%
  set_engine("brulee") %>%
  set_mode("classification")


nn_grid <- grid_regular(learn_rate(), 
                        levels = 4)

nn_rs <- workflow() %>%
  add_recipe(peng_recipe) %>%
  add_model(nn_spec) %>%
  tune_grid(resamples = peng_folds,
            grid = nn_grid,
            metrics = metric_set(accuracy,roc_auc)
  )

final_nn <- finalize_model(nn_spec, select_best(nn_rs, "roc_auc"))

final_nn_fit <- fit(final_nn, island ~ ., peng_train)

write_rds(final_nn_fit,"./models/tuned_peng_nn.rds")