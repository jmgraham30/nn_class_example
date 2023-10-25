library(torch)
library(torchvision)
library(torchdatasets)
library(luz)

set.seed(777)
torch_manual_seed(777)

dir <- "~/.torch-datasets"

train_ds <- tiny_imagenet_dataset(
  dir,
  download = TRUE,
  transform = . %>%
    transform_to_tensor() %>%
    transform_random_affine(
      degrees = c(-30, 30), translate = c(0.2, 0.2)
    ) %>%
    transform_normalize(
      mean = c(0.485, 0.456, 0.406),
      std = c(0.229, 0.224, 0.225)
    )
)

valid_ds <- tiny_imagenet_dataset(
  dir,
  split = "val",
  transform = function(x) {
    x %>%
      transform_to_tensor() %>%
      transform_normalize(
        mean = c(0.485, 0.456, 0.406),
        std = c(0.229, 0.224, 0.225))
  }
)

train_dl <- dataloader(
  train_ds,
  batch_size = 128,
  shuffle = TRUE
)
valid_dl <- dataloader(valid_ds, batch_size = 128)

convnet <- nn_module(
  "convnet",
  initialize = function() {
    self$features <- nn_sequential(
      nn_conv2d(3, 64, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(64, 128, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(128, 256, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(256, 512, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_dropout2d(p = 0.05),
      nn_conv2d(512, 1024, kernel_size = 3, padding = 1), 
      nn_relu(),
      nn_adaptive_avg_pool2d(c(1, 1)),
      nn_dropout2d(p = 0.05),
    )
    self$classifier <- nn_sequential(
      nn_linear(1024, 1024),
      nn_relu(),
      nn_dropout(p = 0.05),
      nn_linear(1024, 1024),
      nn_relu(),
      nn_dropout(p = 0.05),
      nn_linear(1024, 200)
    )
  },
  forward = function(x) {
    x <- self$features(x)$squeeze()
    x <- self$classifier(x)
    x
  }
)

model <- convnet %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    )
  ) 

rates_and_losses <- model %>% lr_finder(train_dl)
rates_and_losses %>% plot()

fitted <- model %>%
  fit(train_dl, epochs = 50, valid_data = valid_dl,
      callbacks = list(
        luz_callback_early_stopping(patience = 2),
        luz_callback_lr_scheduler(
          lr_one_cycle,
          max_lr = 0.01,
          epochs = 50,
          steps_per_epoch = length(train_dl),
          call_on = "on_batch_end"),
        luz_callback_model_checkpoint(path = "cpt_dropout/"),
        luz_callback_csv_logger("logs_dropout.csv")
      ),
      verbose = FALSE)

