library(xtable)
library(kableExtra)
library(knitr)
library(class)
library(ggplot2)
library(MASS)
library(fastDummies)
library(tidyr)
library(dplyr)
#library(tidyverse)
library(ggcorrplot)
library(sampling)
library(pROC)
library(zoo)
library(forecast)
library(Matrix)
library(glmnet)
library(tsutils)

# PART TO SKIP: GENERATES THE FINAL DATASET
df <- read.csv("dataset_finale.csv")
# convert drafted to integer
df$drafted[df$drafted == "False"] <- 0
df$drafted[df$drafted == "True"] <- 1
# round games_played and games_started and height, weight for consistency with other columns
df$games_played <- round(df$games_played, 3)
df$games_started <- round(df$games_started, 3)
df$height <- round(df$height, 3)
df$weight <- round(df$weight, 3)
# make dummy for categorical position
df$drafted <- as.integer(df$drafted)
df <- dummy_cols(df, select_columns = 'position')
# remove useless columns
df =  df %>% select(-c(name, start_season, position))
#drop <- c("name","start_season","position")
# move drafted to last column
df = df %>% relocate(drafted, .after = last_col())

# delete a dummy variable WE DECIDED TO KEEP 3 DUMMIES (we can't because multicollinearity)
df =  df %>% select(-position_Center)
# n features before (not counting end_season):
print(ncol(df))
# remove total rebound since it makes the matrix singular (total reb = def + off reb)
df =  df %>% select(-total_reb)
# (ALREADY TESTED) Removing total points covariates because they cause linear relation
df =  df %>% select(-c(field_goal, field_attmps, field_pct))
# Remove throws that scored points and keep percentage and total throws because collinearity
df =  df %>% select(-c(two_pattamps, three_pattmp, free_attmps))
# remove points because linear combination of the other point throws
df = df %>% select(-c(points))
# remove seasons variable (we decided to keep weight) because consideration on the correlation mtx and boxplots
df =  df %>% select(-c(seasons))
# n features after (not counting end_season):
print(ncol(df))
# IT KEEPS END_SEASON FOR TRAIN TEST SPLIT

# Split dataset in train / test based on end_season
# players per year
print(table(df$end_season))
# drafted per year
table(df$end_season,df$drafted)
#total players
nrow(df)
# info on the test candidate:
ratio = (1437+1474)/15797 * 100
print(ratio)
# make the split
split_season <- 2018
index_test  = which(df$end_season > split_season)
length(index_test)
#index_test
test <- df[index_test, ]
train <- df[-index_test, ]
#remove end_season from both dataframes
train =  train %>% select(-end_season)
test = test %>% select(-end_season)

# NOW WE HAVE THE DATASETS READY TO WORK WITH



# METRICS SECTION THE FUNCTION WE BUILT TO CALCULATE THE METRICS DESCRIBED IN THE REPORT

# First classification metrics function (idea is to show one)
# Confusion matrix
report_confusion_matrix <- function(values, pred) {
  mtx <- as.matrix(table(factor(values,levels = 0:1) , factor(pred, levels =0:1)))
  return (mtx)
}

# accuracy
report_accuracy <- function(values, pred) {
  mtx <- as.matrix(table(factor(values,levels = 0:1) , factor(pred, levels =0:1)))
  accuracy <- round((mtx[1,1] + mtx[2,2]) / sum(mtx), 3)
  return(accuracy)
}
# balanced_accuracy
report_balanced_accuracy <- function(values, pred) {
  recall = report_recall(values, pred)
  specificity = report_specificity(values, pred)
  balanced_accuracy = round((recall+specificity)/2, 3)
  return(balanced_accuracy)
}

# precision
report_precision <- function(values, pred) {
  mtx <- as.matrix(table(factor(values,levels = 0:1) , factor(pred, levels =0:1)))
  precision <-  round(mtx[2, 2] / sum(mtx[, 2]), 3)
  return(precision)
}

# recall
report_recall <- function(values, pred) {
  mtx <- as.matrix(table(factor(values,levels = 0:1) , factor(pred, levels =0:1)))
  recall = round(mtx[2,2] / sum(mtx[2,]), 3)
  return(recall)
}

# specificity 
report_specificity <- function(values, pred) {
  mtx <- as.matrix(table(factor(values,levels = 0:1) , factor(pred, levels =0:1)))
  specificity = round(mtx[1,1] / sum(mtx[1, ]), 3)
  return(specificity)
}

# f1_score
report_f1 <- function(values, pred) {
  precision = report_precision(values, pred)
  recall = report_recall(values, pred)
  f1_score = round(2*((precision*recall)/(precision+recall)), 3)
  return(f1_score)
}

# function to plot roc auc and return the roc (to find threshold)
report_auc <- function(values, pred) {
  metrics.roc <- roc(values, pred, levels=c(0,1))
  # plotting
  plot(metrics.roc, print.auc = TRUE, legacy.axes = TRUE, xlab = 'False positive rate',
       ylab = 'True positive rate', col = 'red')
  return(metrics.roc)
}

# function that returns a vector of all the metrics for a prediction
report_all <- function(model_name,values,pred) {
  # model_name is a string
  # order accuracy, balanced acc, precision, recall, f1
  acc <- report_accuracy(values, pred)
  bal <- report_balanced_accuracy(values, pred)
  pre <- report_precision(values, pred)
  rec <- report_recall(values, pred)
  f1 <- report_f1(values, pred)
  metrics <- list(model_name,acc,bal,pre,rec,f1)
  return(metrics)
}

# Function that prints the table with the performances
metrics_table <- function(report_all_rows) {
  table.df <- data.frame(report_all_rows[[1]],stringsAsFactors = F) #first row
  # remove first row and iterate
  report_all_rows <- report_all_rows[-1]
  for (i in report_all_rows) {
    table.df <- rbind(table.df,i)
  }
  names(table.df) <- c("Models", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1_score")
  return(table.df)
}

# HERE WE SHOW THE PERFORMANCE FOR 2 BASELINE CLASSIFIERS: THE RANDOM ONE AND THE ALL NON DRAFTED ONE (THIS PROVIDES THE BASELINE ACCURACY ON A HIGLY UNBALANCED DATASET LIKE OURS)
set.seed(42)
# a complete random classifier random
pred.random <- sample(c(0,1), replace=TRUE, size=nrow(test))
# a classifier that only sets 0
pred.zeros <- sample(c(0), replace=TRUE, size=nrow(test))
# print with knit the 2 confusion matrix side by side
mtx.random <- report_confusion_matrix(test$drafted, pred.random)
mtx.zeros <- report_confusion_matrix(test$drafted, pred.zeros)
kable(list(mtx.random, mtx.zeros)) # THIS SHOULD PRINT THE 2 MTX SIDE BY SIDE MISSING CAPTIONS (NON SO COME AGGIUNGERLI PER OGNI ELEMENTO DELLA LISTA!!!!!!!!!)
# Print the kable with the metrics
mod.random.metrics <- report_all("Random Classifier", test$drafted, pred.random)
mod.zeros.metrics <- report_all("Zeros Classifier", test$drafted, pred.zeros)
table.df <- metrics_table(list(mod.random.metrics,mod.zeros.metrics))
kable(table.df)


# END OF METRICS SECTION


# START MODELS FOR BINARY CLASSIFICATION SECTION

# FIRST THE GLM

# We have built a function to print the roc-auc curve and return the roc metric to find the best threshold
report_roc <- function(values, pred) {
  metrics.roc <- roc(values, pred, levels=c(0,1))
  # plotting
  plot(metrics.roc, print.auc = TRUE, legacy.axes = TRUE, xlab = 'False positive rate',
       ylab = 'True positive rate', col = 'red')
  # return the roc 
  return(metrics.roc)
}

# model build and summary print to screen
mod.glm <- glm(drafted~., data = train, family = binomial)
summary(mod.glm)
# make the predictions
pred.glm <- predict(mod.glm, test, type= "response")
# find the optimal threshold and print the plot (this also prints the roc)
mod.glm.roc <- report_roc(test$drafted, pred.glm)
# also the default threshold for everything
tr.default <- 0.5 
# print the best threshold
coords(mod.glm.roc, 'best')
tr.glm <- coords(mod.glm.roc, 'best')
# convert predictions in binary to calculate metrics
pred.glm.default <- pred.glm
pred.glm.default[pred.glm.default >= tr.default] <- 1
pred.glm.default[pred.glm.default < tr.default] <- 0
pred.glm.optimal <- pred.glm
pred.glm.optimal[pred.glm.optimal >= tr.glm] <- 1
pred.glm.default[pred.glm.optimal < tr.glm] <- 0

# print this to make an example on how this works
pred.glm.df <- c(pred.glm, test$drafted, pred.glm.default, pred.glm.optimal)
names(pred.glm.df) <- c("Probability", "Ground Truth", "Prediction default T", "Prediction optimal T" )

# print with knit the 2 confusion matrix side by side
mtx.glm.default <- report_confusion_matrix(test$drafted, pred.glm.default)
mtx.glm.optimal <- report_confusion_matrix(test$drafted, pred.glm.optimal)
kable(list(mtx.glm.default, mtx.glm.optimal)) # THIS SHOULD PRINT THE 2 MTX SIDE BY SIDE MISSING CAPTIONS (NON SO COME AGGIUNGERLI PER OGNI ELEMENTO DELLA LISTA!!!!!!!!!)
# Print the kable with the metrics
mod.glm.default.metrics <- report_all("GLM default Threshold",test$drafted, pred.glm.default)
mod.glm.optimal.metrics <- report_all("GLM optimal Threshold",test$drafted, pred.glm.optimal)
table.df <- metrics_table(list(mod.glm.default.metrics, mod.glm.optimal.metrics))
kable(table.df)

# END OF GLM

# START OF LDA

# HERE WE EXCLUDE THE DUMMY VARIABLES
mod.lda <- lda(drafted~.-position_Guard-position_Forward, data = train) # USES MASS LIBRARY
mod.lda # no summary must print this
# make the predictions
pred.lda <- predict(mod.lda, test)
# get probabilities for class "drafted 1"
pred.lda <- pred.lda$posterior[,2]
# find the optimal threshold and print the plot (this also prints the roc)
mod.lda.roc <- report_roc(test$drafted, pred.lda)
# print the best threshold
coords(mod.lda.roc, 'best')
tr.lda <- coords(mod.lda.roc, 'best')
# convert predictions in binary to calculate metrics
pred.lda.default <- pred.lda
pred.lda.default[pred.lda.default >= tr.default] <- 1
pred.lda.default[pred.lda.default < tr.default] <- 0
pred.lda.optimal <- pred.lda
pred.lda.optimal[pred.lda.optimal >= tr.lda] <- 1
pred.lda.default[pred.lda.optimal < tr.lda] <- 0

# print with knit the 2 confusion matrix side by side
mtx.lda.default <- report_confusion_matrix(test$drafted, pred.lda.default)
mtx.lda.optimal <- report_confusion_matrix(test$drafted, pred.lda.optimal)
kable(list(mtx.lda.default, mtx.lda.optimal)) # THIS SHOULD PRINT THE 2 MTX SIDE BY SIDE MISSING CAPTIONS (NON SO COME AGGIUNGERLI PER OGNI ELEMENTO DELLA LISTA!!!!!!!!!)
# Print the kable with the metrics
mod.lda.default.metrics <- report_all("GLM default Threshold",test$drafted, pred.lda.default)
mod.lda.optimal.metrics <- report_all("GLM optimal Threshold",test$drafted, pred.lda.optimal)
table.df <- metrics_table(list(mod.lda.default.metrics, mod.lda.optimal.metrics))
kable(table.df)

# END OF LDA

# START OF QDA

# HERE WE EXCLUDE THE DUMMY VARIABLES
mod.qda <- qda(drafted~.-position_Guard-position_Forward, data = train) # USES MASS LIBRARY
mod.qda # no summary must print this
# make the predictions
pred.qda <- predict(mod.qda, test)
# get probabilities for class "drafted 1"
pred.qda <- pred.qda$posterior[,2]
# find the optimal threshold and print the plot (this also prints the roc)
mod.qda.roc <- report_roc(test$drafted, pred.qda)
# print the best threshold
coords(mod.qda.roc, 'best')
tr.qda <- coords(mod.qda.roc, 'best')
# convert predictions in binary to calculate metrics
pred.qda.default <- pred.qda
pred.qda.default[pred.qda.default >= tr.default] <- 1
pred.qda.default[pred.qda.default < tr.default] <- 0
pred.qda.optimal <- pred.qda
pred.qda.optimal[pred.qda.optimal >= tr.qda] <- 1
pred.qda.default[pred.qda.optimal < tr.qda] <- 0

# print with knit the 2 confusion matrix side by side
mtx.qda.default <- report_confusion_matrix(test$drafted, pred.qda.default)
mtx.qda.optimal <- report_confusion_matrix(test$drafted, pred.qda.optimal)
kable(list(mtx.qda.default, mtx.qda.optimal)) # THIS SHOULD PRINT THE 2 MTX SIDE BY SIDE MISSING CAPTIONS (NON SO COME AGGIUNGERLI PER OGNI ELEMENTO DELLA LISTA!!!!!!!!!)
# Print the kable with the metrics
mod.qda.default.metrics <- report_all("GLM default Threshold",test$drafted, pred.qda.default)
mod.qda.optimal.metrics <- report_all("GLM optimal Threshold",test$drafted, pred.qda.optimal)
table.df <- metrics_table(list(mod.qda.default.metrics, mod.qda.optimal.metrics))
kable(table.df)

# END OF QDA

# START OF KNN

# WE show a function we have made to find the best k on a stratified sample for validation from the training set (same rapporto between classes that is in the training set is forced on the validation set)
# KNN hold out validation (no function for model since the function requires the test set)
search_best_knn <- function(train, colnames_excluded, maxk) {
  # finds out the best performant knn from 1 to maxk with a dedicated test set ( no k fold option )
  # colnames_excluded is a vector of column names to exclude ("drafted" never included)
  # Shuffle the rows
  set.seed(42)
  rows <- sample(nrow(train))
  train <- train[rows, ]
  # get a stratified sampling we want 20% as test
  train.samplesize <- round(0.20*nrow(train))
  # get sample proportions
  zeros <- table(train$drafted)[1]
  ones <- table(train$drafted)[2]
  prop.zeros <- round(zeros / nrow(train) , 3)
  prop.ones <- round(1 - prop.zeros , 3)
  nrows.zeros <- round(train.samplesize * prop.zeros , 0)
  nrows.ones <- train.samplesize - nrows.zeros
  # now we sample these rows from drafted and non drafted
  sample.drafted <- train %>% filter(drafted == 1) %>% slice(1:nrows.ones)
  sample.notdrafted <- train %>% filter(drafted == 0) %>% slice(1:nrows.zeros)
  sample.test <- rbind(sample.drafted, sample.notdrafted)
  # remove the ids from the training set, anti join defaults to all columns
  train <- anti_join(train, sample.test)
  # now we have our training and stratified test set with respect to the classes
  f1 <- c()
  k.values <- seq(1,maxk,1)
  for (i in k.values) {
    knn.pred <- knn(train[-20] %>% select(-colnames_excluded), sample.test[-20] %>% select(-colnames_excluded), cl = train$drafted, k = i)
    f1 <- c(f1, report_f1(sample.test$drafted, knn.pred))
  }
  # PLOT AND BEST OUTSIDE
  mod.knn.info <- list(k.values,f1)
  return(mod.knn.info)
}

# find the best k: using function (maxk=30)
mod.knn.info <- search_best_knn(train, c(), maxk = 30)
mod.knn.k.values <- mod.knn.info[[1]]
mod.knn.f1.values <- mod.knn.info[[2]]

# plot f1 score vs k (should be done in ggplot)
plot(mod.knn.k.values, mod.knn.f1.values, type = 'l', xlab = 'Number of neighbors k', ylab = 'F1 Score')

# print the best value of k
mod.knn.k.best <- mod.knn.k.values[which.max(mod.knn.f1.values)]

# WE WANT TO SHOW THE CONCEPT OF COURSE OF DIMENSIONALITY BY SHOWING 3 PLOTS OF K WITH A PROGRESSIVE SMALLER NUMBER OF FEATURES, SHOWING IT SHIFT TO A HIGHER BEST K (ON THE FULL 19 DIMENSION MODEL IT'S K=1 BECAUSE THEY ARE TOO FAR)
par(mfrow=c(1,3))
#15 features
knn.fifteen <- search_best_knn(train, c("height", "weight", "games_played", "games_started"), maxk = 30)
knn.fifteen.values <- knn.fifteen[[1]]
knn.fifteen.f1.values <- knn.fifteen[[2]]
# plot first 
plot(knn.fifteen.values, knn.fifteen.f1.values, type = 'l', xlab = 'Number of neighbors k', ylab = 'F1 Score')
# 10 features
knn.ten <- search_best_knn(train, c("height", "weight", "games_played", "games_started", "min_per", "two_pointer", "two_pct", "three_ptrs", "three_pct"), maxk = 30)
knn.ten.values <- knn.ten[[1]]
knn.ten.f1.values <- knn.ten[[2]]
# plot second
plot(knn.ten.values, knn.ten.f1.values, type = 'l', xlab = 'Number of neighbors k', ylab = 'F1 Score')
#5 features
knn.five <- search_best_knn(train, c("height", "weight", "games_played", "games_started", "min_per", "two_pointer", "two_pct", "three_ptrs", "three_pct", "free_throws", "free_pct", "assists", "steals", "blocks"), maxk = 30)
knn.five.values <- knn.five[[1]]
knn.five.f1.values <- knn.five[[2]]
# plot third
plot(knn.five.values, knn.five.f1.values, type = 'l', xlab = 'Number of neighbors k', ylab = 'F1 Score')
par(mfrow=c(1,1))

# make the predictions (no model object for knn)
pred.knn <- knn(train[-20], test[-20], cl = train$drafted, k = mod.knn.k.best)

# print with knit the single confusion matrix
mtx.knn <- report_confusion_matrix(test$drafted, pred.knn)
kable(mtx.knn) # MISSING LABEL
# Print the kable with the metrics
mod.knn.metrics <- report_all("KNN with best K",test$drafted, pred.knn)
table.df <- metrics_table(list(mod.knn.metrics))
kable(table.df)

# END OF KNN

### END OF FULL MODELS 

### START OF FEATURE SELECTION ON GLM

# WHAT WE USED: CONSTRAIN BASED STEPWISE SELECTION, AIC BACKWARD STEPWISE SELECTION, BIC BACKWARD STEPWISE SELECTION, GLM LASSO REGULARIZATION
# Unable to use bestglm for aid and bic given that for logistical regression it only exposes a method will exhaustive search, leading to (in our case) 2^19 models evaluation

# CONSTRAIN BASED STEPWISE SELECTION
# start with constrain based stepwise selection (remove the highest p value and refit the model then remove highest p value then refit etc untill all covariates have *** for significance)
# this is the order of the models fitting 
mod.glm.full <- mod.glm <- glm(drafted~., data = train, family = binomial)
mod.glm.red.constrains <- glm(drafted~.-free_pct, data = train, family = binomial)
mod.glm.red.constrains <- glm(drafted~.-free_pct-off_reb, data = train, family = binomial)
mod.glm.red.constrains <- glm(drafted~.-free_pct-off_reb-games_started, data = train, family = binomial)
mod.glm.red.constrains <- glm(drafted~.-free_pct-off_reb-games_started-weight, data = train, family = binomial)
mod.glm.red.constrains <- glm(drafted~.-free_pct-off_reb-games_started-weight-two_pct, data = train, family = binomial)
mod.glm.red.constrains <- glm(drafted~.-free_pct-off_reb-games_started-weight-two_pct-three_pct, data = train, family = binomial)
mod.glm.red.constrains <- glm(drafted~.-free_pct-off_reb-games_started-weight-two_pct-three_pct-def_reb, data = train, family = binomial)
constrains.columns <- c("free_pct", "off_reb", "games_started", "weight", "two_pct", "three_pct", "def_reb")
# print the columns removed and the number of removed
constrains.columns
length(constrains.columns)

# AIC BACKWARD STEPWISE SELECTION
# we SHOW our function for the backward on the glm it returns the list of columns
select_glm_AIC <- function(train) {
  # TESTED : IT WORKS AS INTENDED
  # glm.AIC <- bestglm(train, IC = 'AIC', method = "backward", family=binomial) NON UTILIZZABILE SOLO EXAUSTED CON BINOMIAL (logit link)
  # intializing starting AIC (full model) and covariates list
  aics.best <- c(AIC(glm(drafted~., data = train, family = binomial)))
  columns.kept <- c(names(train[-20]))
  columns.deleted <- c()
  while (TRUE) {
    aics.loop <- c()
    columns.loop <- c()
    for (i in columns.kept) {
      # get all the aics for removing one of the remaining column
      aics.loop <- c (aics.loop, 
                      AIC(glm(drafted~., data = train[,!(names(train) %in% c(columns.deleted, i))], family = binomial)))
      columns.loop <- c(columns.loop, i)
    }
    # break if no improvements
    if(min(aics.loop) > tail(aics.best,1)) {
      break
    }
    # add the best removal to the removed list, this gives the index for columns.loop
    index <- which(aics.loop == min(aics.loop))[1]
    columns.deleted <- c(columns.deleted, columns.loop[index])
    # remove the column removed from the remaining columns list
    columns.kept <- columns.kept[! columns.kept %in% c(columns.loop[index])]
    # add the new AIC value to the best list
    aics.best <- c(aics.best, min(aics.loop))
    # break if we removed all the columns
    if (length(columns.kept) == 0) {
      break
    }
  }
  # return the columns deleted and aic_best 
  aic.columns <- list(aics.best,columns.deleted)
  return(aic.columns)
}

# let's get the deleted columns and the aics to plot
aic.output <- select_glm_AIC(train)
aic.columns <- aic.output[[1]]
aic.values <- aic.output[[2]]
# print columns removed and number
aic.columns
length(aic.columns)
# let's plot the aics (x is the number of columns removed == index of the aic.values vector)
aic.numbers <- c(0:(length(aic.values)-1)) # -1 because it starts from 0
plot(aic.numbers, aic.values, type = 'l', xlab = 'Number of columns removed', ylab = 'best AIC')

# BIC BACKWARD STEPWISE SELECTION
# it's the same function but it uses BIC instead of AIC we don't show the function
select_glm_BIC <- function(train) {
  # intializing starting BIC (full model) and covariate list
  bics.best <- c(BIC(glm(drafted~., data = train, family = binomial)))
  columns.kept <- c(names(train[-20]))
  columns.deleted <- c()
  while (TRUE) {
    bics.loop <- c()
    columns.loop <- c()
    for (i in columns.kept) {
      # get all the bics for removing one of the remaining column
      bics.loop <- c (bics.loop, 
                      BIC(glm(drafted~., data = train[,!(names(train) %in% c(columns.deleted, i))], family = binomial)))
      columns.loop <- c(columns.loop, i)
    }
    # break if no improvements
    if(min(bics.loop) > tail(bics.best,1)) {
      break
    }
    # add the best removal to the removed list, this gives the index for columns.loop
    index = which(bics.loop == min(bics.loop))[1]
    columns.deleted <- c(columns.deleted, columns.loop[index])
    # remove the column removed from the remaining columns list
    columns.kept <- columns.kept[! columns.kept %in% c(columns.loop[index])]
    # add the new AIC value to the best list
    bics.best <- c(bics.best, min(bics.loop))
    # break if we removed all the columns
    if (length(columns.kept) == 0) {
      break
    }
  }
  # return the columns deleted and aic_best 
  bic.columns <- list(bics.best,columns.deleted)
  return(bic.columns)
  
}
# let's get the deleted columns and the aics to plot
bic.output <- select_glm_BIC(train)
bic.columns <- bic.output[[1]]
bic.values <- bic.output[[2]]
# print columns removed and number
bic.columns
length(bic.columns) # bic favors simpler models so columns removed should be more than aic
# let's plot the bics (x is the number of columns removed == index of the bic.values vector)
bic.numbers <- c(0:(length(bic.values)-1)) # -1 because it starts from 0
plot(bic.numbers, bic.values, type = 'l', xlab = 'Number of columns removed', ylab = 'best BIC')

# LASSO
# We used glmnet with 10 fold cross validation to select from 100 values of lambdas
# The values of lambdas to try were pre determines using the function lambdaseq()that finds the max value of lambda that if used sets all coefficients to 0, and then logaritmhically generates other 99 smaller lamdas to try
set.seed(42) # seed for cross validation
x.seq <- as.matrix(train[,1:19])
y.seq <- train[,20]
lambda.seq <- lambdaseq(x.seq, y.seq)$lambda
X.vars <- model.matrix(train$drafted~. , train)[,-1] #removes column of ones from model.matrix that we must use for glmnet
Y.vars <- train$drafted
# uses cross validation 10 fold (default) (uses misclassification error to determine best lambda) to find best lambda over the 100 in the list
cv.lasso <- cv.glmnet(X.vars, Y.vars, lambda = lambda.seq, type.measure = 'class', family ='binomial') 
# print results
cv.lasso$cvm
# get best lambda and print it
best.lambda <- cv.lasso$lambda.min
best.lambda
# we plot the results (default implementation of plot for glmnet)
plot(cv.lasso)
# WE CONSIDER THE LASSO MODEL HERE
# build the model with lasso and best lambda
mod.glm.lasso <- glmnet(X.vars, Y.vars, alpha = 1, lambda = best.lambda, family = 'binomial')
# print coefficients (shows the ones put to zeros)
coef(mod.glm.lasso)

