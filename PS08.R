library(tidyverse)
library(caret)
library(MLmetrics)

# Package for easy timing in R
library(tictoc)



# Demo of timer function --------------------------------------------------
# Run the next 5 lines at once
tic()
Sys.sleep(3)
timer_info <- toc()
runtime <- timer_info$toc - timer_info$tic
runtime



# Get data ----------------------------------------------------------------
# Accelerometer Biometric Competition Kaggle competition data
# https://www.kaggle.com/c/accelerometer-biometric-competition/data
train <- read_csv("~/Downloads/train.csv")

# YOOGE!
dim(train)



# knn modeling ------------------------------------------------------------
model_formula <- as.formula(Device ~ X + Y + Z)

# Values to use:
n_values <- c(1:15000)
k_values <- c(1:200)


runtime_dataframe <- expand.grid(n_values, k_values) %>%
  as_tibble() %>%
  rename(n=Var1, k=Var2) %>%
  mutate(runtime = n*k)
runtime_dataframe



train1 <- train %>% 
  sample_n(15000)

#n set to 15,000, k set to 200 max.

# Time knn here -----------------------------------------------------------

tic()

for (i in 1:200) {
  
  
  model_knn <- caret::knn3(model_formula, data = train1, k = i)
  
  fitted_matrix <- model_knn %>% 
    predict(newdata = train1, type = "prob") %>% 
    round(3)
}

timer_info <- toc()

runtime <- timer_info$toc - timer_info$tic
runtime

fitted_tidy <- fitted_matrix %>% 
  as_tibble()  

# Plot your results ---------------------------------------------------------
# Think of creative ways to improve this barebones plot. Note: you don't have to
# necessarily use geom_point

runtime_plot <- ggplot(runtime_dataframe, aes(x=n, y=k, col=runtime)) +
  geom_line() + ggtitle("Runtime")

runtime_plot
ggsave(filename="firstname_lastname.png", width=16, height = 9)






# Runtime complexity ------------------------------------------------------
# Can you write out the rough Big-O runtime algorithmic complexity as a function
# of:
# -n: number of points in training set
# -k: number of neighbors to consider
# -p: number of predictors used? In this case d is fixed at 3

#Big-O runtime for single instance of knn = (n choose p) (k*n)*p
