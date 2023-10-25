library(tidyverse)
library(tidymodels)
library(keras)
library(torch)
library(luz)

fashion_mnist <- dataset_fashion_mnist()


fashion_mnist_train_x <- fashion_mnist$train$x %>%
  torch_tensor(dtype = torch_float32())


fashion_mnist_train_y <- fashion_mnist$train$y %>%
  torch_tensor(dtype = torch_int16())

train_ds <- tensor_dataset(x=fashion_mnist_train_x,y=fashion_mnist_train_y) 

fashion_mnist_test_x <- fashion_mnist$test$x %>%
  torch_tensor(dtype = torch_float32())

fashion_mnist_test_y <- fashion_mnist$test$y %>%
  torch_tensor(dtype = torch_int16())

test_ds <- tensor_dataset(x=fashion_mnist_test_x,y=fashion_mnist_test_y) 

train_dl <- dataloader(train_ds, batch_size = 1, shuffle = TRUE)

test_dl <- dataloader(test_ds, batch_size = 1, shuffle = TRUE)

