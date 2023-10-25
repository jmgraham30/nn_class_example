library(tidyverse)
library(tidymodels)
library(keras)
library(torch)
library(luz)

fashion_mnist <- dataset_fashion_mnist()


fashion_mnist_train_x <- fashion_mnist$train$x %>%
  torch_tensor()

fashion_mnist_train_y <- fashion_mnist$train$y %>%
  torch_tensor()

tain_ds <- list(x=fashion_mnist_train_x,y=fashion_mnist_train_y)

fashion_mnist_test_x <- fashion_mnist$test$x %>%
  torch_tensor()

fashion_mnist_test_y <- fashion_mnist$test$y %>%
  torch_tensor()

test_ds <- list(x=fashion_mnist_test_x,y=fashion_mnist_test_y)

