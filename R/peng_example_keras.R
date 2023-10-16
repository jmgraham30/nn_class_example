library(tidyverse)
library(tidymodels)
library(keras)
library(ggthemes)

tidymodels_prefer()
theme_set(theme_minimal(base_size = 12))

set.seed(1234)
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

peng_recipe <- recipe(island ~ ., data = peng_train) %>%
  step_normalize(all_numeric_predictors()) 

## A Single MLP

nn_spec <- mlp(hidden_units = 20, epochs = 300, learn_rate = 0.05) %>%
  set_engine("keras") %>%
  set_mode("classification")

peng_wf <- workflow() %>%
  add_recipe(peng_recipe) %>%
  add_model(nn_spec)

peng_nn_fit <- fit(peng_wf,peng_train)

write_rds(peng_nn_fit,"./models/fitted_peng_nn_keras.rds")

peng_nn_fit <- read_rds("models/fitted_peng_nn_keras.rds")

autoplot(peng_nn_fit)

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


predict(peng_nn_fit, peng_test, type = "class") %>% 
  bind_cols(peng_test) %>%
  mutate(correct = case_when(
    island == .pred_class ~ "Correct",
    TRUE ~ "Incorrect"
  )) %>%
  ggplot(aes(x=bill_depth_mm,y=flipper_length_mm,color=correct)) + 
  geom_point() + 
  scale_color_colorblind() + 
  labs(title = "NN Classification")
