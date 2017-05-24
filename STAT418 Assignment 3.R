# install.packages("h2o")

library(h2o)
# h2o.init()
# h2o.shutdown()
h2o.init(nthreads = -1)

# attr(data,"id")
# h2o.ls()
# data <-h2o.assign(data,"H2O_id")
# h2o.rm()

#Imports files into an H2O cloud.
# bank_additional_full.hex_sid_9810_1
# 1) client and H2O cluster are on the same machine -> use h2o.importFile()
# 2) different, load file from local -> use h2o.uploadFile()
# 3) R data frame -> use as.h2o()
path <- "/Users/Rachel/Desktop/STAT418/bank-additional/"
bank <- h2o.importFile(paste0(path,"bank-additional-full.csv"), destination_frame = "bank")
setwd("/Users/Rachel/Desktop/STAT418/SUSY")
susy <- read.csv("SUSY.csv", nrows = 10, header = FALSE)
View(susy)
y <- "y"
x <- setdiff(names(bank), y)

bank_parts <- h2o.splitFrame(bank, c(0.6,0.2))
bank_train <- bank_parts[[1]]
bank_valid <- bank_parts[[2]]
bank_test <- bank_parts[[3]]
rm(bank_parts)




bank_train <- h2o.assign(bank_train,"bank_train")
bank_test <- h2o.assign(bank_test,"bank_test")

## 1- Deep Learning Model
# Build a Deep Neural Network model using CPUs
# Builds a feed-forward multilayer artificial neural network on an H2OFrame
# Upon importing, can define col.names= c(), col.types= c() (numeric versus enum, rather than real, int...)
# Specify destination_frame = "" to replace the same frame, avoiding old versions clogging up the memory
model_DL <- h2o.deeplearning(x, y, bank_train)
predict_DL <- h2o.predict(model_DL, bank_test)

h2o.mse(model_DL)
h2o.confusionMatrix(model_DL)

# h2o:as.data.frame(Downloads the H2O data and then scans it in to an R data frame.
# predict - confidence
as.data.frame(predict_DL)
# comparison
as.data.frame(h2o.cbind(predict_DL$predict, bank_test$y) )
# percentage of correct prediction
mean(predict_DL$predict == bank_test$y)

#Alternative Way to Check Model Performance without Looking into Individual Prediction
h2o.performance(model_DL, bank_test)

# Notice: Randomness, especially randomness in split can lead to perfect score that can not be reproduced

# See H2O Flow at http://127.0.0.1:54321/flow/index.html

# gc() & h2o.rm()
# h2o.rm() -> removes H2O frame
# gc() -> garbage collection, can take place automatically without user intervention, 
# and the primary purpose of calling gc is for the report on memory usage.
# However, it can be useful to call gc after a large object has been removed, 
# as this may prompt R to return memory to the operating system. 
# (Frees up the memory for ones that you no longer have access to.)

# Delete intermediate frames, without leaving too many frames in H2O.(i.e. delete columns)
# Use following:
# data[,2:5] -> intermediate frames created (Every change involves a data copy.)
# attr(data,"id")
# h2o.ls()
# h2o.assign()
# h2o.rm()

# Data Summaries
# h2o.describe(bank)
# h2o.summary(bank)
# h2o.quantile(bank)
# h2o.levels(bank)
# dim(bank)
# nrow(bank)
# ncol(bank)

# Aggregating rows: min, max, mean, mode, sd, ss, sum, and var. nrow() rather than count()
h2o.group_by(bank, by = "y", nrow("y"), mean("age"), mean("duration"))

# Histogram
h2o.hist(bank$age, breaks = 16, plot = TRUE)

# Fetch 1000 rows of age and y from H2O to R
ix <- sort(sample(1:nrow(bank), 1000) )
# as.matrix -> charater matrix because of the factor column "y"
d <- as.data.frame(bank[ix, c("y","age","campaign")])

# Trainning, Validation, Test Split
# 1) Random split (balanced + independent)
# <a> h2o.split() => recommended!!! can end up different sizes
# parts <- h2o.split(data, c(0.6, 0.2) )
# train <- parts[[1]]
# valid <- parts[[2]]
# test <- parts[[3]]
# rm(parts) #Optional
# <b> result in specified sizes, usually for small files. Use only if have good reason to.
# ratios <- c(0.6, 0.2, 0.2)
# sz <- nrow(data)
# indices <- split(1:sz, sample( rep(1:3, sz * ratios) ) )
# train <- data[ indices[[1]], ]
# valid <- data[ indices[[2]], ]
# test <- data[ indices[[3]], ]
# 2) Time split: old-train, later-train, new-test => Times seris (not independent!)
# ratios <- c(0.75, 0.15, 0.1)
# sz <- nrow(data)
# indices <- split(1:sz, rep(1:3, sz * ratios) )
# train <- data[ indices[[1]], ]
# valid <- data[ indices[[2]], ]
# test <- data[ indices[[3]], ]



# Xnames <- names(bank_train)[which(names(bank_train)!="y")]

#LR
system.time({
  model_LR_1 <- h2o.glm(x, y, training_frame = SUSY_train, family = "binomial", alpha = 1.0,
                        lambda_search = TRUE)
})


h2o.auc(h2o.performance(model_LR, SUSY_valid))

model_LR

h2o.auc(h2o.performance(model_LR, SUSY_test))

# Try various values for lambda.
# 
# If setting lambda to a list of test values and use lambda_search = TRUE, the optimal lamdba is choosen by comparing auc of model based on the trainning dataset. However, I want to tune the LR model with different lambda based on validation dataset.
# 
# Therefore, I use gird models to compare
# Example of values to grid over for `lambda`

hyper_params <- list( lambda = c(1, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0) )

# this example uses cartesian grid search because the search space is small
# and we want to see the performance of all models. For a larger search space use
# random grid search instead: list(strategy = "RandomDiscrete")
grid <- h2o.grid("x = x, y = y, family = 'binomial', training_frame = SUSY_train, 
                 validation_frame = SUSY_valid, algorithm = "glm", 
                 grid_id = "SUSY_LR_grid", hyper_params = hyper_params,
                 search_criteria = list(strategy = "Cartesian"))

## Sort the grid models by AUC
sortedGrid <- h2o.getGrid("SUSY_LR_grid", sort_by = "auc", decreasing = TRUE)
sortedGrid


#RF
system.time({
  model_RF <- h2o.randomForest(x, y, training_frame = bank_train, ntrees = 500)
})

h2o.auc(h2o.performance(model_RF, bank_test))

model_RF

#GBM

system.time({
  model_GBM <- h2o.gbm(x, y, training_frame = bank_train, distribution = "bernoulli", 
              ntrees = 300, max_depth = 20, learn_rate = 0.1, 
              nbins = 100, seed = 123)    
})

h2o.auc(h2o.performance(model_GBM, bank_test))

model_GBM