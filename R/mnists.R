library(tidyverse)
library(keras)


hand_digits <- dataset_mnist()

digits <- hand_digits$train$x

sample_image <- as.data.frame(digits[1, , ])
colnames(sample_image) <- seq_len(ncol(sample_image))
sample_image$y <- seq_len(nrow(sample_image))
sample_image <- gather(sample_image, "x", "value", -y)
sample_image$x <- as.integer(sample_image$x)
ggplot(sample_image, aes(x = x, y = y, fill = value)) +
  geom_tile() + scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() + theme_minimal() + 
  theme(panel.grid = element_blank(), legend.position = "none") +
  theme(aspect.ratio = 1) + xlab("") + ylab("")

hand_digits$train$y[1]

fashion_mnist <- dataset_fashion_mnist()

images <- fashion_mnist$train$x

sample_image <- as.data.frame(images[1, , ])
colnames(sample_image) <- seq_len(ncol(sample_image))
sample_image$y <- seq_len(nrow(sample_image))
sample_image <- gather(sample_image, "x", "value", -y)
sample_image$x <- as.integer(sample_image$x)
ggplot(sample_image, aes(x = x, y = y, fill = value)) +
  geom_tile() + scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() + theme_minimal() + 
  theme(panel.grid = element_blank(), legend.position = "none") +
  theme(aspect.ratio = 1) + xlab("") + ylab("")

fashion_mnist$train$y[1]



