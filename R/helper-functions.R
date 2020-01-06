
## -------------------------------------------------------------------##
## Collection of helper functions for Monte Carlo simulation
## Copyright: Hyunjin Song & Petro Tolochko, Dept. of Comm @ Uni Wien ##
## -------------------------------------------------------------------##
require(rstan)
options(mc.cores = parallel::detectCores(logical = F))
expose_stan_functions('functions_stan.stan', show_compiler_warnings = F)

## not in (%!in%)
'%!in%' <- function(x,y)!('%in%'(x,y))

## media content generating function
data.initiate <- function(seed) {

  n_media <- .GlobalEnv$n_media
  n.total <- .GlobalEnv$n.total
  b0 <- .GlobalEnv$b0 ## baseline (unobserved, fixed) effect for media outlet
  b1 <- .GlobalEnv$b1
  b2 <- .GlobalEnv$b2
  b3 <- .GlobalEnv$b3

  N <- 3
  set.seed(seed)
  Sigma <- matrix(rnorm(N*N), N, N)
  Sigma <- Sigma %*% t(Sigma)  # Sigma is PD


  require(MASS)
  set.seed(seed)
  dat <- rmulti_normal(n.total, mu = rnorm(3, 0, 1), ## vector of means
                 Sigma = Sigma)

  media.content.data <- expand.grid(medium = 1:n_media, year = 1:5, day = 1:260, article.id = 1:10) %>%
    as_tibble() %>% arrange(medium, year, day, article.id)
  media.content.data <- cbind(obs.id = 1:n.total, media.content.data) %>% as_tibble()
  set.seed(seed)
  media.content.data <- media.content.data %>%
    mutate(X0 = rep(rnorm(1:n_media), each = max(year) * max(day) * max(article.id)),
           X1 = dat[, 1],
           X2 = dat[, 2],
           X3 = dat[, 3],
           y_true.value =
             ## logit(Y) ~ X1 + X2 + X3 + e
             rbinom(n.total, 1, plogis(X0 + b1*X1 + b2*X2 + b3*X3 + rnorm(n.total))))
  ## X0 = systematic error / rnorm = random error

  ## convert to data.frame object for a faster processing
  setDT(media.content.data); rm(dat)
  media.content.data
}


data.initiate.dict <- function(seed) {

  n_media <- .GlobalEnv$n_media
  n.total <- .GlobalEnv$n.total

  N <- 3
  set.seed(seed)
  Sigma <- matrix(rnorm(N*N), N, N)
  Sigma <- Sigma %*% t(Sigma)  # Sigma is PD

  betas <- c(1, rep(.2, N))

  require(MASS)
  set.seed(seed)
  dat <- rmulti_normal(n.total, mu = rnorm(3, 0, 1), ## vector of means
                       Sigma = Sigma)

  media.content.data <- expand.grid(medium = 1:n_media, year = 1:5, day = 1:260, article.id = 1:10) %>%
    as_tibble() %>% arrange(medium, year, day, article.id)
  media.content.data <- cbind(obs.id = 1:n.total, media.content.data) %>% as_tibble()
  set.seed(seed)
  media.content.data <- media.content.data %>%
    mutate(X0 = rep(rnorm(1:n_media), each = max(year) * max(day) * max(article.id)),
           X1 = round(dat[, 1]),
           X2 = round(dat[, 2]),
           X3 = round(dat[, 3]),
           y_true.value =
             ## logit(Y)
             rbinom(n.total, 1,
                    plogis(X0 + 0.2*X1 + 0.2*X2 + 0.2*X3 + rnorm(n.total))))

  ## convert to data.frame object for a faster processing
  setDT(media.content.data); rm(dat)
  media.content.data
}



## helper for selecting hyperparameter for alpha and beta given desired target kriff.alpha
select.beta.dist.arg <- function(target.k.alpha = c(0.5, 0.7, 0.9)) {
  if (target.k.alpha == 0.5) {alpha = 43; beta = 7}
  #else if (target.k.alpha == 0.6) {alpha = 48; beta = 6}
  else if (target.k.alpha == 0.7) {alpha = 50; beta = 4}
  #else if (target.k.alpha == 0.8) {alpha = 60; beta = 3}
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
                                 k = c(2, 5, 10),
                                 target.k.alpha = c(0.5, 0.7, 0.9)) {

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

## ------------------------------------- ##
## Step 2-1. Training SML (for SML only) ##
## ------------------------------------- ##

train.SML <- function(seed, data) {
    set.seed(seed)
    dat <- data[sort(sample(1:n.total, size = 5000, replace = F)), ]
    y <- dat[, y_true.value] %>% as.matrix %>% as.numeric %>% as.factor
    dat[, y_true.value := y]

    library("caret")
    model <- train(y_true.value ~ X1 + X2 + X3, dat,
                   method = "bayesglm",
                   trControl = trainControl(method = "cv",
                                            number = 10,
                                            verboseIter = FALSE))

    model ## model for prediction given new observation
    ## initial prediction accuracy is stored in model$results$accuracy
}

## ---------------------------------------------------------------- ##
## Step 2-2. Code additional data independently for validation test ##
## ---------------------------------------------------------------- ##

# once establish acceptable reliability, procede to additional coding for validation
# determine the "(imperfect) gold standard"

code.valiation.data <- function(seed,
                                data = media.content.data,
                                k = c(2, 5, 10),
                                n.units = c(650, 1300, 6500, 13000),
                                target.k.alpha = c(0.5, 0.7, 0.9),
                                random.sample = c(1, 0),
                                duplicated.coding = c(1, 0)) {

  ## k = no. of raters
  ## randomly sample "n.units" number of obs per each k coder
  if (random.sample == 1) {
    set.seed(seed)
    dat <- data[sort(sample(1:n.total, size = n.units, replace = F)), ]

  } else if (random.sample == 0) {
    ## non-random sample
    set.seed(seed)
    ran.start <- sample(1:117000, 1)
    dat <- data[ran.start:(ran.start + n.units - 1), ]
  } else stop()

  true.val <- dat[, y_true.value]
  alpha.val <- select.beta.dist.arg(target.k.alpha)[1]
  beta.val <- select.beta.dist.arg(target.k.alpha)[2]

  n.units.per.coder <- n.units/k

  if (duplicated.coding == 0) {

    ## sole-coding per each coder
    dat.validation <- dat[, human.coding := unlist(lapply(1:k, function(i) {

      ## copy vector of true values for appropriate range and pass to human code function
      range <- seq(from = (1+n.units.per.coder*(i-1)), to = (n.units.per.coder*(i)))
      observer.rating <- human_choice_rng(s = n.units.per.coder,
                                          S = true.val[range],
                                          a = alpha.val,
                                          b = beta.val,
                                          get_rng(seed = seed), get_stream())
      observer.rating
    })) ## end of unlist
    ] ## end of data.table

  } else if (duplicated.coding == 1) {

    k.codings <- paste0("human.coding.", 1:k)
    set.seed(seed)
    dat.validation <- dat[, eval(k.codings) := lapply(1:k, function(g = 1) {
                              human_choice_rng(s = n.units,
                                               S = dat[, y_true.value],
                                               a = alpha.val,
                                               b = beta.val,
                                               get_rng(seed = seed), get_stream())})]
    dat.validation[, human.coding := rowMeans(.SD) %>% round,
                     .SDcols = grep("^human.coding.[:digit:]*",
                                    colnames(dat.validation), value = T)]

  } else stop()

  return(dat.validation)

}


## -------------------------------- ##
## Step 3. Main simulation function ##
## -------------------------------- ##

sim_study <- function(seed,
                      k = c(2, 5, 10),
                      n.units = c(650, 1300, 6500, 13000),
                      target.k.alpha = c(0.5, 0.7, 0.9),
                      random.sample = c(1, 0),
                      duplicated.coding = c(1, 0)) {

  data <- data.initiate(seed)

  ## train model
  trained.model <- train.SML(seed, data)

  ## code validation dataset
  ## set.seed(12345)
  dat.validation <- code.valiation.data(seed = seed, data = data, k = k, n.units = n.units,
                                        target.k.alpha = target.k.alpha,
                                        random.sample = random.sample,
                                        duplicated.coding = duplicated.coding)

  ## machine prediction from trained model
  dat.validation[, m.predicted := predict(trained.model, dat.validation)]

  ## calculate prediction performance based on machine predictions vs. human coding
  require(caret)
  dat.validation$m.predicted <- factor(dat.validation$m.predicted)
  dat.validation$human.coding <- factor(dat.validation$human.coding)

  ## save some parameters
  obs.accuracy <- sum(diag(dat.validation[, table(m.predicted, human.coding) / .N]))
  obs.precision <- dat.validation[, posPredValue(m.predicted, human.coding, positive = "1")]
  obs.recall <- dat.validation[, sensitivity(m.predicted, human.coding, positive = "1")]
  if (is.na(obs.precision)) obs.precision <- 0
  if (is.na(obs.recall)) obs.recall <- 0
  obs.F1 <- (2 * obs.precision * obs.recall) / (obs.precision + obs.recall)

  ## now, scale up the machine coding rule to entire media data
  data[, m.predicted := predict(trained.model, data)]
  data$m.predicted <- factor(data$m.predicted)
  data$y_true.value <- factor(data$y_true.value)

  ## calculate overall machine prediction performance against unknown true gold standard
  ## measure of overlap with "m.predicted" and "true (yet, hypothetically, unknown) value"
  true.accuracy <- sum(diag(data[, table(m.predicted, y_true.value) / .N]))
  true.precision <- data[, posPredValue(m.predicted, y_true.value, positive = "1")]
  true.recall <- data[, sensitivity(m.predicted, y_true.value, positive = "1")]
  true.F1 <- (2 * true.precision * true.recall) / (true.precision + true.recall)


  ## gather metrics to report
  results = data.frame(
    obs.accuracy = obs.accuracy,
    obs.precision = obs.precision,
    obs.recall = obs.recall,
    obs.F1 = obs.F1,

    prevalence = data[, table(y_true.value)/.N][2],

    true.accuracy = true.accuracy,
    true.precision = true.precision,
    true.recall = true.recall,
    true.F1 = true.F1)

  results
}


# Dictionary ------------------------------------------------------------

sim_study.dict <- function(seed,
                           k = c(2, 5, 10),
                           n.units = c(650, 1300, 6500, 13000),
                           target.k.alpha = c(0.5, 0.7, 0.9),
                           random.sample = c(1, 0),
                           duplicated.coding = c(1, 0)) {

  # features <- 5
  # betas <- c(rep(.2, features))
  # neg_draws <- rdirichlet(1, c(rep(1.5, 6), rep(1, 5)))
  # pos_draws <- rdirichlet(1, c(rep(1, 5), rep(1.5, 6)))
  # data <- data.initiate.dict(features = features, pos_draws = pos_draws, neg_draws = neg_draws)

  data <- data.initiate.dict(seed)

  dat.validation <- code.valiation.data(seed, data = data, k = k, n.units = n.units,
                                        target.k.alpha = target.k.alpha,
                                        random.sample = random.sample,
                                        duplicated.coding = duplicated.coding)

  ## dictionary rules -- assumes mean of features
  dat.validation[, dict.predicted := ifelse(rowMeans(.SD) > 0, 1, 0), .SDcols = c("X1", "X2", "X3")]

  ## calculate prediction performance based on dictionary predictions vs. human coding
  require(caret)
  dat.validation$dict.predicted <- factor(dat.validation$dict.predicted)
  dat.validation$human.coding <- factor(dat.validation$human.coding)

  ## save some parameters
  obs.accuracy <- sum(diag(dat.validation[, table(dict.predicted, human.coding) / .N]))
  obs.precision <- dat.validation[, posPredValue(dict.predicted, human.coding, positive = "1")]
  obs.recall <- dat.validation[, sensitivity(dict.predicted, human.coding, positive = "1")]
  if (is.na(obs.precision)) obs.precision <- 0
  if (is.na(obs.recall)) obs.recall <- 0
  obs.F1 <- (2 * obs.precision * obs.recall) / (obs.precision + obs.recall)

  ## now, scale up the machine coding rule to entire media data
  data[, dict.predicted := ifelse(rowMeans(.SD) > 0, 1, 0), .SDcols = c("X1", "X2", "X3")]
  data$dict.predicted <- factor(data$dict.predicted)
  data$y_true.value <- factor(data$y_true.value)

  ## calculate overall machine prediction performance against unknown true gold standard
  ## measure of overlap with "m.predicted" and "true (yet, hypothetically, unknown) value"
  true.accuracy <- sum(diag(data[, table(dict.predicted, y_true.value) / .N]))
  true.precision <- data[, posPredValue(dict.predicted, y_true.value, positive = "1")]
  true.recall <- data[, sensitivity(dict.predicted, y_true.value, positive = "1")]
  true.F1 <- (2 * true.precision * true.recall) / (true.precision + true.recall)


  ## gather metrics to report
  results = data.frame(
    obs.accuracy = obs.accuracy,
    obs.precision = obs.precision,
    obs.recall = obs.recall,
    obs.F1 = obs.F1,

    prevalence = data[, table(y_true.value)/.N][2],

    true.accuracy = true.accuracy,
    true.precision = true.precision,
    true.recall = true.recall,
    true.F1 = true.F1)

  results
}


## main simulation function

expand.grid.df <- function(...) Reduce(function(...) merge(..., by=NULL), list(...))

sim.all.scenario <- function() {
  cond <- expand.grid.df(
    tibble(k = c(2, 5, 10)),
    tibble(n.units = c(600, 1300, 6500, 13000)),
    tibble(target.k.alpha = c(0.5, 0.7, 0.9)),
    tibble(random.sample = c(1, 0)), ## 1 = TRUE, 0 = FALSE
    tibble(duplicated.coding = c(1, 0)) ## 1 = TRUE, 0 = FALSE
  ) %>% setDT(.)

  ## 1000 resamples
  cond <- cond[rep(seq_len(nrow(cond)), 1000), ]
  N <- cond[, .N]

  cond <- cbind(seed = 1:N, cond)

  RNGkind("L'Ecuyer-CMRG")
  set.seed(12345)
  require(pbmcapply)
  est.list <- pbmclapply(1:N, function(x) {
            est <- do.call(sim_study, as.list(cond[x, ]))
            est
          }, mc.cores = 10) %>% do.call(rbind, .)

  out <- cbind(cond, est.list) %>% as_tibble() %>% setDT()
  out
}


sim.all.scenario_dict <- function() {
  cond <- expand.grid.df(
    tibble(k = c(2, 5, 10)),
    tibble(n.units = c(600, 1300, 6500, 13000)),
    tibble(target.k.alpha = c(0.5, 0.7, 0.9)),
    tibble(random.sample = c(1, 0)), ## 1 = TRUE, 0 = FALSE
    tibble(duplicated.coding = c(1, 0)) ## 1 = TRUE, 0 = FALSE
  ) %>% setDT(.)

  ## 1000 resamples
  cond <- cond[rep(seq_len(nrow(cond)), 1000), ]
  N <- cond[, .N]

  cond <- cbind(seed = 1:N, cond)

  require(pbmcapply)
  est.list <- pbmclapply(1:N, function(x) {
    est <- do.call(sim_study.dict, as.list(cond[x, ]))
    est
  }, mc.cores = 10) %>% do.call(rbind, .)

  out <- cbind(cond, est.list) %>% as_tibble() %>% setDT()
  out
}


## find 95% CIs of differences in two MAPEs
boot.diff <- function(data, i, grouping_by) {
  resample.data <- data[i, ]
  setDT(resample.data)
  MAPEs <- resample.data[, .(MAPE = mean(MAPE, na.rm = T)), by = grouping_by][, MAPE]
  diff(MAPEs)
}


# # ## use bite compiler to gain some speed
 require(compiler)
 enableJIT(3)

# ## bite-compile helper functions
 data.initiate <- cmpfun(data.initiate)
 data.initiate.dict <- cmpfun(data.initiate.dict)
 select.beta.dist.arg <- cmpfun(select.beta.dist.arg)
 reliability.training <- cmpfun(reliability.training)
 average.agreement <- cmpfun(average.agreement)
 code.valiation.data <- cmpfun(code.valiation.data)
 sim_study <- cmpfun(sim_study)
 sim_study.dict <- cmpfun(sim_study.dict)
 sim.all.scenario <- cmpfun(sim.all.scenario)
 sim.all.scenario_dict <- cmpfun(sim.all.scenario_dict)
