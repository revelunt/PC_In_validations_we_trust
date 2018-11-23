
## -------------------------------------------------------------------##
## Monte Carlo Simulation code for paper: XXXX                        ##
##                                                                    ##
## Copyright: Hyunjin Song & Petro Tolochko, Dept. of Comm @ Uni Wien ##
## -------------------------------------------------------------------##


## install required libraries if not already installed
list.of.packages <- c("data.table", "tidyverse", "dplyr",
                      "ggplot2", "grid", "stringi", "tidyr",
                      "irr", "caret", "rstudioapi",
                      'naivebayes', 'brms', 'rstan')
new.packages <- list.of.packages[!(list.of.packages %in%
                                     installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, dependencies = T)

## automatically setting working directories
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(brms)
library(ggplot2)
library(grid)
library(stringi)
library(tidyr)
library(dplyr)
library(parallel)
library(data.table)
library(rstan)
library(caret)
library(tidytext)
library(MCMCpack)
library(extraDistr)
theme_set(theme_bw())

## --------------------------------------------- ##
## Step 1. Set up the entire media contents data ##
## --------------------------------------------- ##

n_days = 365*20 # timespan = 20 years
n_media = 10 # 10 media outlets
n_articles_per_day = 10 # 10 political news per day per outlet

n.total <- n_days*n_media*n_articles_per_day
n.total # total number of media contents to be coded = 365*20*10*10 = 730000

## we assume following data-generating process of dichotomous Y,
## which is unknown to a researcher
## Y ~ rbinom(n.total, 1, inv.logit(b0 + b1*X1 + b2*X2 + b3*X3))
## X1, X2, and X3 are the features of the text
## (e.g., frame, valence, visibility, etc) that are assumed to be
## normally distributed with their means equal to mu1 to mu3
## with some correlations among those document features (see below).
## While those text features are known to a researcher,
## population "parameter values" of true data-generating process are
## assumed to be unknown to a researcher.
## We set following parameter values for true data generating model:
## b0 = .2, b1 = .5, b2 = .2, b3 = .6

## ture population parameters
b0 = 0; b1 = .5; b2 = .2; b3 = .6

## source helper files
source("helper-functions.R")

# media.content.data <- data.initiate()
# media.content.data[, table(y_true.value)/.N]


## -------------------------------------------------------------------##
## Step 2. randomly sample data for reliability test for human coding ##
## -------------------------------------------------------------------##

# human.coding.data <- media.content.data[sort(sample(1:n.total, size = 100, replace = F)), ]
#
# ## calcurate kripp.alpha for the hand-coding data
# human.coding.data <- reliability.training(dat = human.coding.data, k = 4, target.k.alpha = 0.7)
#
# require(irr)
# human.coding.data[, .SD, .SDcols = c("obs.with.error.1", "obs.with.error.2",
#                                      "obs.with.error.3", "obs.with.error.4")] %>%
#                    t() %>% kripp.alpha(method = "nominal")
#
# human.coding.data[, .SD, .SDcols = c("obs.with.error.1", "obs.with.error.2",
#                                      "obs.with.error.3", "obs.with.error.4")] %>% average.agreement()
#
# ## cf. distributions of kripp.alpha from 1,000 random samples
# ## long-run properties closely match with target K-alpha value
#
# reliability.target.test <- lapply(1:1000, function(i) {
#   test <- reliability.training(dat = media.content.data[sort(sample(1:n.total, size = 100, replace = F)), ],
#                                k = 4, target.k.alpha = 0.7)
#   test <- test[, .SD, .SDcols = paste("obs.with.error", 1:4, sep = ".")]
#   ka <- kripp.alpha(t(test), method = "nominal")
#   agree <- average.agreement(test)
#   return(data.frame(ka = ka$value, agree = agree))
# }); apply(do.call(rbind, reliability.target.test), 2, mean)

## ---------------------------------------------------------------------------------------------- ##
## Step 3. Simulate machine coding rule for validation test data given precision/recall parameter ##
## ---------------------------------------------------------------------------------------------- ##

## based on "human.coding" we attempt to come up with machine-based prediction of y
## using some document features
require(parallel)
setwd('..'); setwd('./Results')

# Naive Bayes Simulation --------------------------------------------------
## set seed for reproducibility
RNGkind("L'Ecuyer-CMRG")
set.seed(12345)

sim.naive.results <- mclapply(1:1000, function(k) {
  out <- sim.all.scenario.once_naive()
  out
  }, mc.cores = parallel::detectCores(logical = T))
sim.naive.results <- do.call("rbind", sim.naive.results)
sim.naive.results[, replication := rep(1:1000, each = 80)]
save(sim.naive.results, file = "sim.naive.results.Rdata")
rm(sim.naive.results); gc()


# Binomial Regression Simulation ------------------------------------------
## set seed for reproducibility
RNGkind("L'Ecuyer-CMRG")
set.seed(12345)
sim.binomial.results <- mclapply(1:1000, function(k) {
  out <- sim.all.scenario.once_binomial()
  out
  }, mc.cores = parallel::detectCores(logical = T))
sim.binomial.results <- do.call("rbind", sim.binomial.results)
sim.binomial.results[, replication := rep(1:1000, each = 80)]
save(sim.binomial.results, file = "sim.binomial.results.Rdata")
rm(sim.binomial.results); gc()


# Bag-of-words Simulation -------------------------------------------------
## set seed for reproducibility
RNGkind("L'Ecuyer-CMRG")
set.seed(12345)
sim.bow.results <- mclapply(1:1000, function(k) {
  out <- sim.all.scenario.once_bow()
  out
  }, mc.cores = parallel::detectCores(logical = T))
sim.bow.results <- do.call("rbind", sim.bow.results)
sim.bow.results[, replication := rep(1:1000, each = 80)]
save(sim.bow.results, file = "sim.bow.results.Rdata")
rm(sim.bow.results); gc()


## ------------------- ##
## summarizing results ##
## ------------------- ##

require(patchwork)
sim.naive.results[, n.units_f := factor(n.units, levels = c(50, 100, 250, 500),
                     labels = c("Annotation N = 50", "Annotation N = 100",
                                 "Annotation N = 250", "Annotation N = 500"))]

p1 <- ggplot(sim.naive.results, aes(x = target.k.alpha, y = accuracy.overall)) +
  geom_smooth(method = "lm", alpha = 0.2, color = "black") + theme_bw() +
  facet_grid( ~ n.units_f) +
  xlab("Target Kripp alpha values") + ylab("Overall Accuracy (against true value)") +
  theme(legend.position="bottom")

p2 <- ggplot(sim.naive.results, aes(x = target.k.alpha, y = f.overall)) +
  geom_smooth(method = "lm", alpha = 0.2, color = "black") + theme_bw() +
  facet_grid( ~ n.units_f) +
  xlab("Target Kripp alpha values") + ylab("Overall F1 score (using true value)") +
  theme(legend.position="bottom")

pdf("naive.bayes.summary.01.pdf", width = 12, height = 10, paper = "a4r")
p1 + p2 + plot_layout(nrow = 2)
dev.off()


## prevalence-adjusted bias (using regression models)
sim.naive.results[, abs.bias.accuracy := abs((Valdat.accuracy/accuracy.overall) - 1)]
sim.naive.results[, abs.bias.F1 := abs((Valdat.f/f.overall) - 1)]

p3 <- ggplot(sim.naive.results,
             aes(x = target.k.alpha, y = abs.bias.accuracy, color = factor(k))) +
  geom_smooth(method = "lm", alpha = 0.2, aes(color = factor(k))) + theme_bw() +
  facet_grid( ~ n.units_f) +
  xlab("Target Kripp alpha values") + ylab("Abs Bias of Accuracy (validation vs. true value)") +
  theme(legend.position="none") +
  guides(color = guide_legend(title = "No of coders"))

p4 <- ggplot(sim.naive.results,
             aes(x = target.k.alpha, y = abs.bias.F1, color = factor(k))) +
  geom_smooth(method = "lm", alpha = 0.2, aes(color = factor(k))) + theme_bw() +
  facet_grid( ~ n.units_f) +
  xlab("Target Kripp alpha values") + ylab("Abs Bias of F1 (validation vs. true value)") +
  theme(legend.position="bottom") +
  guides(color = guide_legend(title = "No of coders"))

pdf("naive.bayes.summary.02.pdf", width = 12, height = 10, paper = "a4r")
p3 + p4 + plot_layout(nrow = 2)
dev.off()


## alternatively,
sim.naive.results[, bias.accuracy := (Valdat.accuracy - accuracy.overall)/accuracy.overall]
sim.naive.results[, bias.F1 := (Valdat.f - f.overall)/f.overall]

p3_1 <- sim.naive.results[, .(bias.accuracy = median(bias.accuracy),
                      lwr = quantile(bias.accuracy, 0.16, na.rm = T),
                      upr = quantile(bias.accuracy, 0.84, na.rm = T)),
                  by = c("k", "target.k.alpha", "n.units_f")] %>%
  ggplot(., aes(y = bias.accuracy, x = factor(k), color = factor(target.k.alpha))) +
  geom_point(position = position_dodge(0.7)) +
  geom_errorbar(aes(ymin = lwr, ymax = upr), position = position_dodge(0.7)) +
  geom_hline(yintercept = 0, color = "grey", linetype = 2) +
  facet_grid( ~ n.units_f) +
  xlab("k = No. of coders") + ylab("% Bias in Accuracy (validation vs. true value)") +
  theme(legend.position="none") +
  guides(color = guide_legend(title = "Target Kripp alpha values"))

p4_1 <- sim.naive.results[, .(bias.F1 = median(bias.F1),
                              lwr = quantile(bias.F1, 0.16, na.rm = T),
                              upr = quantile(bias.F1, 0.84, na.rm = T)),
                          by = c("k", "target.k.alpha", "n.units_f")] %>%
  ggplot(., aes(y = bias.F1, x = factor(k), color = factor(target.k.alpha))) +
  geom_point(position = position_dodge(0.7)) +
  geom_errorbar(aes(ymin = lwr, ymax = upr), position = position_dodge(0.7)) +
  geom_hline(yintercept = 0, color = "grey", linetype = 2) +
  facet_grid( ~ n.units_f) +
  xlab("k = No. of coders") + ylab("% Bias in F1 (validation vs. true value)") +
  theme(legend.position="bottom") +
  guides(color = guide_legend(title = "Target Kripp alpha values"))

pdf("naive.bayes.summary.03.pdf", width = 12, height = 10, paper = "a4r")
p3_1 + p4_1 + plot_layout(nrow = 2)
dev.off()
