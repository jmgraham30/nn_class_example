library(tidyverse)
library(tidymodels)
library(brulee)
library(ggthemes)
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

glimpse(penguins)

penguins %>%
  ggplot(aes(x=bill_depth_mm,y=flipper_length_mm,color=island)) + 
  geom_point() + 
  scale_color_colorblind() +
  labs(color="Island")

peng_split <- initial_split(penguins, strata = island)
peng_train <- training(peng_split)
peng_test <- testing(peng_split)

peng_folds <- vfold_cv(peng_train, strata = island, v = 5)

peng_recipe <- recipe(island ~ ., data = peng_train) %>%
  step_normalize(all_numeric_predictors()) 

## A Single MLP

nn_spec <- mlp(hidden_units = 10, epochs = 150L) %>%
  set_engine("brulee") %>%
  set_mode("classification")

peng_wf <- workflow() %>%
  add_recipe(peng_recipe) %>%
  add_model(nn_spec)

peng_nn_fit <- fit(peng_wf,peng_train)

#write_rds(peng_nn_fit,"./models/fitted_peng_nn.rds")

#peng_nn_fit <- read_rds("./models/fitted_peng_nn.rds")

predict(peng_nn_fit, peng_test, type = "class") %>% 
  bind_cols(peng_test) %>% 
  conf_mat(island,.pred_class)

predict(peng_nn_fit, peng_test, type = "prob") %>% 
  bind_cols(peng_test) %>% 
  roc_auc(island, .pred_Biscoe:.pred_Torgersen)


predict(peng_nn_fit, peng_test, type = "prob") %>% 
  bind_cols(peng_test) %>% 
  roc_curve(island, .pred_Biscoe:.pred_Torgersen) %>%
  ggplot(aes(x=specificity,y=sensitivity,color=.level)) + 
  geom_path(linewidth=1) + 
  scale_color_colorblind() +
  scale_x_reverse()


# A Tuned MLP

#nn_spec <- mlp(hidden_units = 10,
#               epochs = 150L,
#               learn_rate = tune()) %>%
#  set_engine("brulee") %>%
#  set_mode("classification")


#nn_grid <- grid_regular(learn_rate(), 
#                          levels = 4)

#nn_rs <- workflow() %>%
#  add_recipe(peng_recipe) %>%
#  add_model(nn_spec) %>%
#  tune_grid(resamples = peng_folds,
#            grid = nn_grid,
#            metrics = metric_set(accuracy,roc_auc)
#  )

#autoplot(nn_rs)

#final_nn <- finalize_model(nn_spec, select_best(nn_rs, "roc_auc"))

#final_nn_fit <- fit(final_nn, island ~ ., peng_train)


################peng_nn_fit_tuned <- read_rds("./models/tuned_peng_nn.rds")

peng_nn_pred <- peng_nn_fit_tuned %>%
  predict(peng_test,type = "class") 


peng_nn_pred %>%
  bind_cols(peng_test %>% select(island)) %>%
  conf_mat(island,.pred_class)

peng_test %>%
  bind_cols(predict(peng_nn_fit_tuned, peng_test,type="prob")) %>%
  roc_auc(island,.pred_Biscoe:.pred_Torgersen) 


peng_test %>%
  bind_cols(predict(peng_nn_fit_tuned, peng_test,type="prob")) %>%
  roc_curve(island,.pred_Biscoe:.pred_Torgersen) %>%
  ggplot(aes(x=specificity,y=sensitivity,color=.level)) + 
  geom_path(linewidth=1) + 
  scale_x_reverse() + 
  scale_color_colorblind() +
  labs(color = "Island")


######### Tuned Decision Tree for Comparison

tree_spec <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("classification")

tree_grid <- grid_regular(cost_complexity(), 
                          tree_depth(), 
                          min_n(), 
                          levels = 4)

tree_rs <- workflow() %>%
  add_recipe(peng_recipe) %>%
  add_model(tree_spec) %>%
  tune_grid(resamples = peng_folds,
            grid = tree_grid,
            metrics = metric_set(accuracy,roc_auc)
  )


final_tree <- finalize_model(tree_spec, select_best(tree_rs, "roc_auc"))

final_tree_fit <- fit(final_tree, island ~ ., peng_train)


peng_tree_pred <- final_tree_fit %>%
  predict(peng_test,type = "class") 


peng_tree_pred %>%
  bind_cols(peng_test %>% select(island)) %>%
  conf_mat(island,.pred_class)

peng_test %>%
  bind_cols(predict(final_tree_fit, peng_test,type="prob")) %>%
  roc_auc(island,.pred_Biscoe:.pred_Torgersen) 


peng_test %>%
  bind_cols(predict(final_tree_fit, peng_test,type="prob")) %>%
  roc_curve(island,.pred_Biscoe:.pred_Torgersen) %>%
  ggplot(aes(x=specificity,y=sensitivity,color=.level)) + 
  geom_path(linewidth=1) + 
  scale_x_reverse() + 
  scale_color_colorblind() +
  labs(color = "Island")


