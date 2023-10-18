library(tidyverse)
library(ISLR2)
library(torch)
library(luz) # high-level interface for torch
library(torchvision) # for datasets and image transformation
library(torchdatasets) # for datasets we are going to use
library(zeallot)

theme_set(theme_minimal(base_size = 12))

device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"

torch_manual_seed(13)


Gitters <- na.omit(Hitters)
n <- nrow(Gitters)
set.seed(13)
ntest <- trunc(n / 3)
testid <- sample(1:n, ntest)

modnn <- nn_module(
  initialize = function(input_size) {
    self$hidden <- nn_linear(input_size, 50)
    self$activation <- nn_relu()
    self$dropout <- nn_dropout(0.4)
    self$output <- nn_linear(50, 1)
  },
  forward = function(x) {
    x %>% 
      self$hidden() %>% 
      self$activation() %>% 
      self$dropout() %>% 
      self$output()
  }
)

x <- model.matrix(Salary ~ . - 1, data = Gitters) %>% scale()
y <- Gitters$Salary

modnn <- modnn %>% 
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_rmsprop,
    metrics = list(luz_metric_mae())
  ) %>% 
  set_hparams(input_size = ncol(x))

fitted <- modnn %>% 
  fit(
    data = list(x[-testid, ], matrix(y[-testid], ncol = 1)),
    valid_data = list(x[testid, ], matrix(y[testid], ncol = 1)),
    epochs = 50
  )

plot(fitted)

npred <- predict(fitted, x[testid, ])

class(npred)

npred$device

npred$shape

y_pred <- torch_tensor(y[testid])$to(device = "mps")$reshape(npred$shape)

class(y_pred)

y_pred$device

y_pred$shape

torch_subtract(y_pred, npred)$abs()$mean()
