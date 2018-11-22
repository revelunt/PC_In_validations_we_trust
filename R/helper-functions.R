
## -------------------------------------------------------------------##
## Collection of helper functions for Monte Carlo simulation
## Copyright: Hyunjin Song & Petro Tolochko, Dept. of Comm @ Uni Wien ##
## -------------------------------------------------------------------##
require(rstan)
options(mc.cores = parallel::detectCores())
expose_stan_functions('functions_stan.stan')

## media content generating function
data.initiate <- function() {

  n_media <- .GlobalEnv$n_media
  n.total <- .GlobalEnv$n.total
  b0 <- .GlobalEnv$b0
  b1 <- .GlobalEnv$b1
  b2 <- .GlobalEnv$b2
  b3 <- .GlobalEnv$b3

  N <- 3
  Sigma <- matrix(rnorm(N*N), N, N)
  Sigma <- Sigma %*% t(Sigma)  # Sigma is PD


  require(MASS)
  dat <- rmulti_normal(n.total, mu = rnorm(3, 0, 1), ## vector of means
                 Sigma = Sigma)

  media.content.data <- expand.grid(year = 1:20, day = 1:365, medium = 1:n_media, article.id = 1:10) %>% as_tibble() %>% arrange(year, day)
  media.content.data <- cbind(obs.id = 1:n.total, media.content.data) %>% as_tibble() %>%
    mutate(X1 = dat[, 1],
           X2 = dat[, 2],
           X3 = dat[, 3],
           y_true.value =
             ## logit(Y) ~ X1 + X2 + X3 + e
             rbinom(n.total, 1, plogis(b0 + b1*X1 + b2*X2 + b3*X3 + rnorm(n.total))))

  ## convert to data.frame object for a faster processing
  setDT(media.content.data); rm(dat)
  media.content.data
}

data.initiate.bow <- function(features = 10) {

  betas <- c(rep(.2, features))

  n_media <- .GlobalEnv$n_media
  n.total <- .GlobalEnv$n.total

  media.content.data <- expand.grid(year = 1:20, day = 1:365, medium = 1:n_media, article.id = 1:10) %>% as_tibble() %>% arrange(year, day)
  media.content.data <- cbind(obs.id = 1:n.total, media.content.data) %>% as_tibble()


  dat <- t(sapply(1:n.total, function(x) f_1(features)))
  dat <- dat %>%
    as_tibble() %>%
    mutate(y_true.value = rbinom(n.total, 1, plogis(dat %*% betas[1:features] + rnorm(n.total))))

  dat <- dat[, c(sample(1:10, 3), 11)]

  full_data <- cbind(media.content.data, dat)

  setDT(full_data)

  return(full_data)
}


## helper for selecting hyperparameter for alpha and beta given desired target kriff.alpha
select.beta.dist.arg <- function(target.k.alpha = c(0.5, 0.6, 0.7, 0.8, 0.9)) {
  if (target.k.alpha == 0.5) {alpha = 43; beta = 7}
  else if (target.k.alpha == 0.6) {alpha = 48; beta = 6}
  else if (target.k.alpha == 0.7) {alpha = 50; beta = 4}
  else if (target.k.alpha == 0.8) {alpha = 60; beta = 3}
  else if (target.k.alpha == 0.9) {alpha = 40; beta = 1}
  else stop(print("please select valid target.k.alpha values!"))
  c(alpha = alpha, beta = beta)
}

## -------------------------------------------------------------------##
## Step 2. randomly sample data for reliability test for human coding ##
## -------------------------------------------------------------------##

## Backward-generate k vectors of human judgements given target value of k.alpha
## this assumes binary or dichotomous judgements, k human coders, no missing data.
reliability.training <- function(dat = human.coding.data,
                                 k = c(2, 4, 7, 10),
                                 target.k.alpha = c(0.5, 0.6, 0.7, 0.8, 0.9)) {

  ## k = no. of raters
  ## dat = supplied dataset (subset of entire media content data)

  ## n of cases for human coding
  setDT(dat)
  n.units <- dat[, .N]
  alpha.val <- select.beta.dist.arg(target.k.alpha)[1][[1]]
  beta.val <- select.beta.dist.arg(target.k.alpha)[2][[1]]

  v_length = n.units

  observer.ratings.all <- sapply(1:k, function(g = 1) {

    observer.rating <- human_choice_rng(s = n.units,
                                             S = dat[, y_true.value],
                                             a = alpha.val,
                                             b = beta.val)
    observer.rating
  })

  colnames(observer.ratings.all) <- paste("obs.with.error", 1:k, sep = ".")
  dat <- cbind(dat, observer.ratings.all)
  setDT(dat)
  return(dat)

}


average.agreement <- function(dat) {
  cols <- colnames(dat)
  combn.length <- dim(combn(cols, m = 2L))[2]

  holsti.agree <- sapply(seq(combn.length), function(i) {
    confusion.mat <- dat[, table(.SD), .SDcols = (combn(cols, m = 2L)[, i])]
    holsti.agree <- sum(diag(confusion.mat))/sum(confusion.mat)
    holsti.agree
  })
  ## return mean holsti agreement across all pairs of coders
  mean(holsti.agree)
}



## ---------------------------------------------------------------- ##
## Step 2-2. Code additional data independently for validation test ##
## ---------------------------------------------------------------- ##

# once establish acceptable reliability, procede to additional coding for validation
# determine the "(imperfect) gold standard" for "n.units" no. of data, coded separately by n no. of coders
code.valiation.data <- function(data = media.content.data,
                                k = c(2, 4, 7, 10),
                                n.units = c(50, 100, 250, 500),
                                target.k.alpha = c(0.5, 0.6, 0.7, 0.8, 0.9)) {

  ## k = no. of raters
  ## randomly sample "n.units" number of obs per each k coder
  dat <- data[sort(sample(1:n.total, size = n.units*k, replace = F)), ]
  true.val <- dat[, y_true.value]
  alpha.val <- select.beta.dist.arg(target.k.alpha)[1]
  beta.val <- select.beta.dist.arg(target.k.alpha)[2]

  dat.validation <- dat[, human.coding := unlist(lapply(1:k, function(i) {
    ## copy vector of true values for appropriate range and pass to human code function
    range <- seq(from = (1+n.units*(i-1)), to = (n.units*(i)))
    #observer.rating <- sapply(true.val[range], function(x) human_code(x, alpha = alpha.val, beta = beta.val))
    observer.rating <- human_choice_rng(s = n.units,
                                        S = true.val[range],
                                        a = alpha.val,
                                        b = beta.val)
    observer.rating
  })) ## end of unlist
  ] ## end of data.table

  dat.validation

}


## -------------------------------- ##
## Step 3. Main simulation function ##
## -------------------------------- ##

sim_study <- function(k = c(2, 4, 7, 10),
                      n.units = c(50, 100, 250, 500),
                      target.k.alpha = c(0.5, 0.6, 0.7, 0.8, 0.9)) {

  data <- data.initiate()
  ## set.seed(12345)
  dat.validation <- code.valiation.data(data = data, k = k, n.units = n.units,
                                        target.k.alpha = target.k.alpha)


  dat.validation[ , human.coding := as.factor(as.numeric(human.coding))]

  library(naivebayes)
  test.fit <- naive_bayes(human.coding ~ X1 + X2 + X3, usekernel = T,
                          data = dat.validation)


  dat.validation[, predict := predict(test.fit, type = "prob")[,2]]

  require(ROCR)
  coding.performance <- dat.validation[, performance(prediction(predict, human.coding), "f")]
  f.values <- coding.performance@y.values[[1]]
  max.f.values <- max(f.values[is.finite(f.values)])

  condition <- coding.performance@y.values[[1]] == max.f.values & !is.na(coding.performance@y.values[[1]])
  prob.cutoff <- coding.performance@x.values[[1]][condition]
  prob.cutoff <- prob.cutoff[1]

  ## make final binary predictions based on predicted prob against prob.cutoff
  ## this algorithm constitues the machine coding rule
  ## that later scales up to the entire corpus
  dat.validation[, binary.prediction := ifelse(predict >= prob.cutoff, 1, 0)]

  ## calculate prediction performance based on final binary predictions
  require(caret)
  dat.validation$binary.prediction <- factor(dat.validation$binary.prediction)
  dat.validation$human.coding <- factor(dat.validation$human.coding)

  ## save some parameters
  obs.accuracy <- sum(diag(dat.validation[, table(binary.prediction, human.coding) / .N]))
  obs.precision <- dat.validation[, posPredValue(binary.prediction, human.coding, positive = "1")]
  obs.recall <- dat.validation[, sensitivity(binary.prediction, human.coding, positive = "1")]
  obs.f.val <- (2 * obs.precision * obs.recall) / (obs.precision + obs.recall)

  ## now, scale up the machine coding rule to entire media data
  ml.prediction <- as_tibble(data[, .(y_true.value = y_true.value)]) %>%
    mutate(ml.prediction = predict(test.fit, newdata = data, type = "prob")[,2]) %>%
    mutate(ml.prediction = ifelse(ml.prediction > prob.cutoff, 1, 0)) %>% setDT()


  ## calculate overall machine prediction performance against unknown true gold standard
  ## measure of overlap with "ml.prediction" and "true (yet, hypothetically, unknown) value"
  accuracy.overall <- sum(diag(ml.prediction[, table(y_true.value,ml.prediction)/.N]))
  precision.overall <- posPredValue(factor(ml.prediction$ml.prediction), factor(ml.prediction$y_true.value), positive = "1")
  recall.overall <- sensitivity(factor(ml.prediction$ml.prediction), factor(ml.prediction$y_true.value), positive = "1")

  outcome <- ml.prediction[, performance(prediction(ml.prediction, y_true.value), "f")]
  F1.overall <- outcome@y.values[[1]][2]

  ## gather metrics to report
  results = data.frame(
    prob.cutoff = prob.cutoff,
    Valdat.accuracy = obs.accuracy,
    Valdat.precision = obs.precision,
    Valdat.recall = obs.recall,
    Valdat.f = obs.f.val,
    prevalence = data[, table(y_true.value)/.N][2],
    accuracy.overall = accuracy.overall,
    precision.overall = precision.overall,
    recall.overall = recall.overall,
    f.overall = F1.overall)
  results
}





# Binomial logit ----------------------------------------------------------

sim_study_binomial <- function(k = c(2, 4, 7, 10),
                      n.units = c(50, 100, 250, 500),
                      target.k.alpha = c(0.5, 0.6, 0.7, 0.8, 0.9)) {

  data <- data.initiate()
  ## set.seed(12345)
  dat.validation <- code.valiation.data(data = data, k = k, n.units = n.units,
                                        target.k.alpha = target.k.alpha)

  ## use binary logistic regression to predict the case assignment
  # test.fit <- glm(human.coding ~ X1 + X2 + X3, family = binomial(), data = dat.validation)

  dat.validation[ , human.coding := as.factor(as.numeric(human.coding))]

  test.fit <- glm(y_true.value ~ X1 + X2 + X3, data = dat.validation, family = binomial(link = "logit"))


  dat.validation[, predict := predict(test.fit, type = "response")]

  require(ROCR)
  coding.performance <- dat.validation[, performance(prediction(predict, human.coding), "f")]
  f.values <- coding.performance@y.values[[1]]
  max.f.values <- max(f.values[is.finite(f.values)])

  condition <- coding.performance@y.values[[1]] == max.f.values & !is.na(coding.performance@y.values[[1]])
  prob.cutoff <- coding.performance@x.values[[1]][condition]
  prob.cutoff <- prob.cutoff[1]

  ## make final binary predictions based on predicted prob against prob.cutoff
  ## this algorithm constitues the machine coding rule
  ## that later scales up to the entire corpus
  dat.validation[, binary.prediction := ifelse(predict >= prob.cutoff, 1, 0)]

  ## calculate prediction performance based on final binary predictions
  require(caret)
  dat.validation$binary.prediction <- factor(dat.validation$binary.prediction)
  dat.validation$human.coding <- factor(dat.validation$human.coding)

  ## save some parameters
  obs.accuracy <- sum(diag(dat.validation[, table(binary.prediction, human.coding) / .N]))
  obs.precision <- dat.validation[, posPredValue(binary.prediction, human.coding, positive = "1")]
  obs.recall <- dat.validation[, sensitivity(binary.prediction, human.coding, positive = "1")]
  obs.f.val <- (2 * obs.precision * obs.recall) / (obs.precision + obs.recall)

  ## now, scale up the machine coding rule to entire media data
  ml.prediction <- as_tibble(data[, .(y_true.value = y_true.value)]) %>%
    mutate(ml.prediction = predict(test.fit, newdata = data, type = "response")) %>%
    mutate(ml.prediction = ifelse(ml.prediction > prob.cutoff, 1, 0)) %>% setDT()


  ## calculate overall machine prediction performance against unknown true gold standard
  ## measure of overlap with "ml.prediction" and "true (yet, hypothetically, unknown) value"
  accuracy.overall <- sum(diag(ml.prediction[, table(y_true.value,ml.prediction)/.N]))
  precision.overall <- posPredValue(factor(ml.prediction$ml.prediction), factor(ml.prediction$y_true.value), positive = "1")
  recall.overall <- sensitivity(factor(ml.prediction$ml.prediction), factor(ml.prediction$y_true.value), positive = "1")

  outcome <- ml.prediction[, performance(prediction(ml.prediction, y_true.value), "f")]
  F1.overall <- outcome@y.values[[1]][2]

  ## gather metrics to report
  results = data.frame(
    prob.cutoff = prob.cutoff,
    Valdat.accuracy = obs.accuracy,
    Valdat.precision = obs.precision,
    Valdat.recall = obs.recall,
    Valdat.f = obs.f.val,
    prevalence = data[, table(y_true.value)/.N][2],
    accuracy.overall = accuracy.overall,
    precision.overall = precision.overall,
    recall.overall = recall.overall,
    f.overall = F1.overall)
  results
}


# SVM ---------------------------------------------------------------------


sim_study_svm <- function(k = c(2, 4, 7, 10),
                      n.units = c(50, 100, 250, 500),
                      target.k.alpha = c(0.5, 0.6, 0.7, 0.8, 0.9)) {

  data <- data.initiate()
  ## set.seed(12345)
  dat.validation <- code.valiation.data(data = data, k = k, n.units = n.units,
                                        target.k.alpha = target.k.alpha)

  ## use binary logistic regression to predict the case assignment
  # test.fit <- glm(human.coding ~ X1 + X2 + X3, family = binomial(), data = dat.validation)

  # ## use bootstrapping for find some better values
  # require(boot)
  # test.coefs <- boot(data = dat.validation, function(dat, i) {
  #   fit.update <- update(test.fit, data = dat[i, ])
  #   coef(fit.update)
  # }, 1000)

  # coefs <- apply(test.coefs$t, 2, median)

  dat.validation[ , human.coding := as.factor(as.numeric(human.coding))]

  #library(naivebayes)
  library(e1071)
  test.fit <- svm(y_true.value ~ X1 + X2 + X3, data = dat.validation)

  # coefs <- coef(test.fit)
  # predictions <- with(dat.validation, inv.logit(coefs[1] + coefs[2]*X1 + coefs[3]*X2 + coefs[4]*X3))

  dat.validation[, predict := predict(test.fit)]

  require(ROCR)
  coding.performance <- dat.validation[, performance(prediction(predict, human.coding), "f")]
  f.values <- coding.performance@y.values[[1]]
  max.f.values <- max(f.values[is.finite(f.values)])

  condition <- coding.performance@y.values[[1]] == max.f.values & !is.na(coding.performance@y.values[[1]])
  prob.cutoff <- coding.performance@x.values[[1]][condition]
  prob.cutoff <- prob.cutoff[1]

  ## make final binary predictions based on predicted prob against prob.cutoff
  ## this algorithm constitues the machine coding rule
  ## that later scales up to the entire corpus
  dat.validation[, binary.prediction := ifelse(predict >= prob.cutoff, 1, 0)]

  ## calculate prediction performance based on final binary predictions
  require(caret)
  dat.validation$binary.prediction <- factor(dat.validation$binary.prediction)
  dat.validation$human.coding <- factor(dat.validation$human.coding)

  ## save some parameters
  obs.accuracy <- sum(diag(dat.validation[, table(binary.prediction, human.coding) / .N]))
  obs.precision <- dat.validation[, posPredValue(binary.prediction, human.coding, positive = "1")]
  obs.recall <- dat.validation[, sensitivity(binary.prediction, human.coding, positive = "1")]
  obs.f.val <- (2 * obs.precision * obs.recall) / (obs.precision + obs.recall)

  ## now, scale up the machine coding rule to entire media data
  ml.prediction <- as_tibble(data[, .(y_true.value = y_true.value)]) %>%
    mutate(ml.prediction = predict(test.fit, newdata = data)) %>%
    mutate(ml.prediction = ifelse(ml.prediction > prob.cutoff, 1, 0)) %>% setDT()


  ## calculate overall machine prediction performance against unknown true gold standard
  ## measure of overlap with "ml.prediction" and "true (yet, hypothetically, unknown) value"
  accuracy.overall <- sum(diag(ml.prediction[, table(y_true.value,ml.prediction)/.N]))
  precision.overall <- posPredValue(factor(ml.prediction$ml.prediction), factor(ml.prediction$y_true.value), positive = "1")
  recall.overall <- sensitivity(factor(ml.prediction$ml.prediction), factor(ml.prediction$y_true.value), positive = "1")

  outcome <- ml.prediction[, performance(prediction(ml.prediction, y_true.value), "f")]
  F1.overall <- outcome@y.values[[1]][2]

  ## gather metrics to report
  results = data.frame(
    prob.cutoff = prob.cutoff,
    Valdat.accuracy = obs.accuracy,
    Valdat.precision = obs.precision,
    Valdat.recall = obs.recall,
    Valdat.f = obs.f.val,
    prevalence = data[, table(y_true.value)/.N][2],
    accuracy.overall = accuracy.overall,
    precision.overall = precision.overall,
    recall.overall = recall.overall,
    f.overall = F1.overall)
  results
}










# Bag of words ------------------------------------------------------------

betas <- c(rep(.2, 10))

# Dirichlet priors on the categorical distribution
f_1 <- function(x){
  neg_draws <- rdirichlet(1, c(rep(1.5, 6), rep(1, 5)))
  pos_draws <- rdirichlet(1, c(rep(1, 5), rep(1.5, 6)))
  test <- rbinom(1, 1, .5)
  if (test == 1) {
    obs <- replicate(x, rcat(1, pos_draws)) - 6
  } else {obs <- replicate(x, rcat(1, neg_draws)) - 6}

  return(obs)
}



sim_study.bow <- function(k = c(2, 4, 7, 10),
                          n.units = c(50, 100, 250, 500),
                          target.k.alpha = c(0.5, 0.6, 0.7, 0.8, 0.9)) {
  features = 10

  data <- data.initiate.bow(features)

  dat.validation <- code.valiation.data(data = data, k = k, n.units = n.units,
                                        target.k.alpha = target.k.alpha)


  dat.validation[ , human.coding := as.factor(as.numeric(human.coding))]

  dat.validation[, predict := ifelse(dat.validation[, 6] + dat.validation[, 7] + dat.validation[, 8] > 0, 1, 0)]

  require(ROCR)
  coding.performance <- dat.validation[, performance(prediction(predict, human.coding), "f")]
  f.values <- coding.performance@y.values[[1]]
  max.f.values <- max(f.values[is.finite(f.values)])

  condition <- coding.performance@y.values[[1]] == max.f.values & !is.na(coding.performance@y.values[[1]])
  prob.cutoff <- coding.performance@x.values[[1]][condition]
  prob.cutoff <- prob.cutoff[1]


  ## Predictions from bag-of-words are already binary, so just duplicate the vector
  dat.validation[, binary.prediction := predict]

  ## calculate prediction performance based on final binary predictions
  require(caret)
  dat.validation$binary.prediction <- factor(dat.validation$binary.prediction)
  dat.validation$human.coding <- factor(dat.validation$human.coding)

  ## save some parameters
  obs.accuracy <- sum(diag(dat.validation[, table(binary.prediction, human.coding) / .N]))
  obs.precision <- dat.validation[, posPredValue(binary.prediction, human.coding, positive = "1")]
  obs.recall <- dat.validation[, sensitivity(binary.prediction, human.coding, positive = "1")]
  obs.f.val <- (2 * obs.precision * obs.recall) / (obs.precision + obs.recall)

  ## now, scale up the machine coding rule to entire media data
  ml.prediction <- as_tibble(data[, .(y_true.value = y_true.value)]) %>%
    mutate(ml.prediction = ifelse(data[, 6] + data[, 7] + data[, 8] > 0, 1, 0)) %>%
    #mutate(ml.prediction = ifelse(ml.prediction > prob.cutoff, 1, 0)) %>%
    setDT()


  ## calculate overall machine prediction performance against unknown true gold standard
  ## measure of overlap with "ml.prediction" and "true (yet, hypothetically, unknown) value"
  accuracy.overall <- sum(diag(ml.prediction[, table(y_true.value,ml.prediction)/.N]))
  precision.overall <- posPredValue(factor(ml.prediction$ml.prediction), factor(ml.prediction$y_true.value), positive = "1")
  recall.overall <- sensitivity(factor(ml.prediction$ml.prediction), factor(ml.prediction$y_true.value), positive = "1")

  outcome <- ml.prediction[, performance(prediction(ml.prediction, y_true.value), "f")]
  F1.overall <- outcome@y.values[[1]][2]

  ## gather metrics to report
  results = data.frame(
    prob.cutoff = prob.cutoff,
    Valdat.accuracy = obs.accuracy,
    Valdat.precision = obs.precision,
    Valdat.recall = obs.recall,
    Valdat.f = obs.f.val,
    prevalence = data[, table(y_true.value)/.N][2],
    accuracy.overall = accuracy.overall,
    precision.overall = precision.overall,
    recall.overall = recall.overall,
    f.overall = F1.overall)
  results
}




## use bite compiler to gain some speed
require(compiler)
enableJIT(3)

## bite-compile helper functions
data.initiate <- cmpfun(data.initiate)
data.initiate.bow <- cmpfun(data.initiate.bow)
select.beta.dist.arg <- cmpfun(select.beta.dist.arg)
reliability.training <- cmpfun(reliability.training)
average.agreement <- cmpfun(average.agreement)
code.valiation.data <- cmpfun(code.valiation.data)
sim_study <- cmpfun(sim_study)
sim_study_binomial <- cmpfun(sim_study_binomial)
sim_study_svm <- cmpfun(sim_study_svm)
sim_study.bow <- cmpfun(sim_study.bow)
