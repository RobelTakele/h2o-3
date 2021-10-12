setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")

# add test from Erin Ledell
glmBetaConstraints <- function() {
  df <- h2o.importFile("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")
  test <- h2o.importFile("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")

  y <- "response"
  x <- setdiff(names(df), y)
  df[,y] <- as.factor(df[,y])
  test[,y] <- as.factor(test[,y])
  
  # Split off a validation_frame
  ss <- h2o.splitFrame(df, seed = 1)
  train <- ss[[1]]
  valid <- ss[[2]]
  
  # Some comparisons
  m0 <- h2o.glm(x = x, y = y, training_frame = train, validation_frame = valid, family = "binomial")
  m1 <- h2o.glm(x = x, y = y, training_frame = train, validation_frame = valid, family = "binomial", lambda_search = TRUE)
  m2 <- h2o.glm(x = x, y = y, training_frame = train, validation_frame = valid, family = "binomial", non_negative = TRUE)
  m3 <- h2o.glm(x = x, y = y, training_frame = train, validation_frame = valid, family = "binomial", non_negative = TRUE, lambda_search = TRUE)
  m4 <- h2o.glm(x = x, y = y, training_frame = train, family = "binomial")
  m5 <- h2o.glm(x = x, y = y, training_frame = train, family = "binomial", lambda_search = TRUE)
  m6 <- h2o.glm(x = x, y = y, training_frame = train, family = "binomial", non_negative = TRUE)
  m7 <- h2o.glm(x = x, y = y, training_frame = train, family = "binomial", non_negative = TRUE, lambda_search = TRUE)
  
  models <- c(m0, m1, m2, m3, m4, m5, m6)
  
  for (m in models) {
    cat(sprintf("validation_frame: %s\n", m@parameters$validation_frame))
    cat(sprintf("lambda_search: %s\n", m@parameters$lambda_search))
    cat(sprintf("non_negative: %s\n", m@parameters$non_negative))
    cat("-------------------------\n")
    cat(sprintf("Test AUC: %f\n", h2o.auc(h2o.performance(m, test))))
    cat(sprintf("Test Logloss: %f\n", h2o.logloss(h2o.performance(m, test))))
    cat(sprintf(
      "Test Res Deviance: %f\n\n",
      h2o.residual_deviance(h2o.performance(m, test))
    ))
  } 
  
  brower()
  # in terms of performance, m0 should be worse than m1
  print("done")
}

doTest("GLM: Compare GLM with and without beta constraints", glmBetaConstraints)
