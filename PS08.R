library(plyr)
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
n_values <- c(1000, 3000, 7500, 10000, 30000, 75000, 100000, 175000, 250000, 300000)
k_values <- c(10, 30, 60, 80, 100)

n_vec <- c (rep(1000, 5), rep(3000, 5), rep(7500, 5), rep(10000, 5), rep(30000, 5), 
            rep(75000, 5), rep(100000, 5), rep(175000, 5), rep(250000, 5), rep(300000, 5)) 





# Time knn here -----------------------------------------------------------

#Max value of n that R will run for this loop is around 500,000

PS_runtime = list()
for (j in n_values) {
  trainPS <- train %>% 
    sample_n(j)
  
  for (i in k_values) {
    tic()
    
    model_knn <- caret::knn3(model_formula, data = trainPS, k = i)
    
    fitted_matrix <- model_knn %>% 
      predict(newdata = trainPS, type = "prob") %>% 
      round(3)
    timer_info <- toc()
    runtime <- timer_info$toc - timer_info$tic
    PS_runtime[[length(PS_runtime)+1]] = runtime
  }
  
}

Runtime_frame <- ldply (PS_runtime, data.frame)

Runtime_frame <- Runtime_frame %>% 
  select(Runtime = X..i..) %>% 
  mutate(k = rep(k_values, 10)) %>% 
  mutate(n = n_vec)



# Plot your results ---------------------------------------------------------
# Think of creative ways to improve this barebones plot. Note: you don't have to
# necessarily use geom_point

PS_runtime_plot <- ggplot(Runtime_frame, aes(x = Runtime_frame$n, y = Runtime_frame$Runtime, col = k)) + 
  geom_point() +geom_smooth() + ggtitle("Runtime Plot") + xlab("n values") + ylab("Runtime (in seconds)")

PS_runtime_plot
ggsave(filename="Kiryu_Kawahata.png", width=16, height = 9)






# Runtime complexity ------------------------------------------------------
# Can you write out the rough Big-O runtime algorithmic complexity as a function
# of:
# -n: number of points in training set
# -k: number of neighbors to consider
# -p: number of predictors used? In this case p is fixed at 3

#Big-O runtime for single instance of knn = (n*p)+k

#The runtime increases as the value of n increases. n seems to have the most influence over runtime.
#The value for k seems to influence the runtime to a lesser degree, and so I made k an added vaue for the O notation.
#Interestingly runtime for different values of k for a given n don't always follow the same pattern. 
#Using less neighbors sometimes ran longer than a knn with more neighbors. I admittedly am not entirely sure the exact role p predictors
#plays in the big-O for runtime. 