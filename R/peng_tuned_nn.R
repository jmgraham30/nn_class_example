library(tidyverse)
library(tidymodels)
library(brulee)

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

peng_folds <- vfold_cv(peng_train, strata = island, v = 10)

peng_recipe <- recipe(island ~ ., data = peng_train) %>%
  step_normalize(all_numeric_predictors()) 

# A Tuned MLP

nn_spec <- mlp(hidden_units = tune(),
               epochs = 150,
               learn_rate = tune()) %>%
  set_engine("brulee") %>%
  set_mode("classification")


nn_grid <- grid_regular(hidden_units(),
                        learn_rate(), 
                        levels = 8)

nn_rs <- workflow() %>%
  add_recipe(peng_recipe) %>%
  add_model(nn_spec) %>%
  tune_grid(resamples = peng_folds,
            grid = nn_grid,
            metrics = metric_set(accuracy,roc_auc)
  )

p_res <- nn_rs %>%
  unnest(.metrics) %>%
  filter(.metric == "roc_auc") %>%
  group_by(hidden_units,learn_rate) %>%
  summarise(mean=mean(.estimate)) %>%
  ggplot(aes(x=learn_rate,y=mean,color=factor(hidden_units))) + 
  geom_point() + 
  geom_line() + 
  scale_x_log10() + 
  labs(x="Learning rate",y="ROC AUC",color="Hidden \n units")

ggsave("plots/nn_tune_res.png")
