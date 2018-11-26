
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

# ----------- Naive Bayes ----------- #

## ---------------- ##
## Basic desciptive ##
## ---------------- ##

if (!("sim.naive.results" %in% ls())) load("sim.naive.results.Rdata")
sim.naive.results[, n.units_f := factor(n.units, levels = c(50, 100, 250, 500),
                     labels = c("Annotation N = 50", "Annotation N = 100",
                                 "Annotation N = 250", "Annotation N = 500"))]
sim.naive.results[, k_f := factor(k, levels = c(2, 4, 7, 10),
                                  labels = c("Coder \nn = 2", "Coder \nn = 4",
                                             "Coder \nn = 7", "Coder \nn = 10"))]

## overall accuracy as a fuction of target.k.alpha
plot0 <- sim.naive.results[, .(accuracy = median(accuracy.overall, na.rm = T),
                               lwr = quantile(accuracy.overall, 0.025, na.rm = T),
                               upr = quantile(accuracy.overall, 0.975, na.rm = T)),
                           by = c("k_f", "target.k.alpha", "n.units_f")]

p0 <- ggplot(plot0, aes(x = accuracy, y = target.k.alpha, xmin = lwr, xmax = upr)) +
  geom_point(size = 1.5) + geom_errorbarh(height = 0) +
  xlab("") + ylab("Target K alpha values") +
  ggtitle("Overall Accuracy: Naive Bayes") +
  geom_vline(xintercept = sim.naive.results[, mean(accuracy.overall)],
             color = "gray", linetype = 2) + ## reference line is overall mean
  theme_bw() + theme(legend.position="none", plot.title = element_text(hjust = 0.5)) +
  facet_grid(k_f ~ n.units_f)

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

pdf("BAYES_summary_01.pdf", width = 12, height = 10, paper = "a4r")
p1 + ggtitle("Overall Classification Quality: Naive Bayes") +
  theme(plot.title = element_text(hjust = 0.5)) + p2 + plot_layout(nrow = 2)
dev.off()


## relative bias
sim.naive.results[, abs.bias.accuracy := abs((Valdat.accuracy/accuracy.overall) - 1)]
sim.naive.results[, abs.bias.F1 := abs((Valdat.f/f.overall) - 1)]

# p3 <- ggplot(sim.naive.results,
#              aes(x = target.k.alpha, y = abs.bias.accuracy, color = factor(k))) +
#   geom_smooth(method = "lm", alpha = 0.2, aes(color = factor(k))) + theme_bw() +
#   facet_grid( ~ n.units_f) +
#   xlab("Target Kripp alpha values") + ylab("Abs Bias of Accuracy (validation vs. true value)") +
#   theme(legend.position="none") +
#   guides(color = guide_legend(title = "No of coders"))
#
# p4 <- ggplot(sim.naive.results,
#              aes(x = target.k.alpha, y = abs.bias.F1, color = factor(k))) +
#   geom_smooth(method = "lm", alpha = 0.2, aes(color = factor(k))) + theme_bw() +
#   facet_grid( ~ n.units_f) +
#   xlab("Target Kripp alpha values") + ylab("Abs Bias of F1 (validation vs. true value)") +
#   theme(legend.position="bottom") +
#   guides(color = guide_legend(title = "No of coders"))
#
# pdf("BAYES_summary_02.pdf", width = 12, height = 10, paper = "a4r")
# p3 + ggtitle("Absolute Degree of Bias Against True Values: Naive Bayes") +
#   theme(plot.title = element_text(hjust = 0.5)) + p4 + plot_layout(nrow = 2)
# dev.off()


## alternatively,
sim.naive.results[, bias.accuracy := (Valdat.accuracy/accuracy.overall)]
sim.naive.results[, bias.F1 := (Valdat.f/f.overall)]

p3_1 <- sim.naive.results[, .(bias.accuracy = median(bias.accuracy),
                      lwr = quantile(bias.accuracy, 0.025, na.rm = T),
                      upr = quantile(bias.accuracy, 0.975, na.rm = T)),
                  by = c("k", "target.k.alpha", "n.units_f")] %>%
  ggplot(., aes(y = bias.accuracy, x = factor(k), color = factor(target.k.alpha))) +
  geom_point(position = position_dodge(0.7)) +
  geom_errorbar(aes(ymin = lwr, ymax = upr), position = position_dodge(0.7)) +
  geom_hline(yintercept = 1, color = "grey", linetype = 2) +
  facet_grid( ~ n.units_f) +
  xlab("k = No. of coders") + ylab("Relative Bias in Accuracy (validation vs. true value)") +
  theme(legend.position="none") +
  guides(color = guide_legend(title = "Target Kripp alpha values"))

p4_1 <- sim.naive.results[, .(bias.F1 = median(bias.F1),
                              lwr = quantile(bias.F1, 0.025, na.rm = T),
                              upr = quantile(bias.F1, 0.975, na.rm = T)),
                          by = c("k", "target.k.alpha", "n.units_f")] %>%
  ggplot(., aes(y = bias.F1, x = factor(k), color = factor(target.k.alpha))) +
  geom_point(position = position_dodge(0.7)) +
  geom_errorbar(aes(ymin = lwr, ymax = upr), position = position_dodge(0.7)) +
  geom_hline(yintercept = 1, color = "grey", linetype = 2) +
  facet_grid( ~ n.units_f) +
  xlab("k = No. of coders") + ylab("Relative Bias in F1 (validation vs. true value)") +
  theme(legend.position="bottom") +
  guides(color = guide_legend(title = "Target Kripp alpha values"))

# pdf("BAYES_summary_03.pdf", width = 12, height = 10, paper = "a4r")
# p3_1 + ggtitle("Relative Bias Against True Values: Naive Bayes") +
#   theme(plot.title = element_text(hjust = 0.5)) + p4_1 + plot_layout(nrow = 2)
# dev.off()


## ---------------------------------------------------------- ##
## Amaong those studies which pass the validation check!      ##
## cutoff value is the mean of accuracy/F1 score from Study 1 ##
## ---------------------------------------------------------- ##

# sim.naive.results[, results := ifelse(Valdat.accuracy > 0.6487,
#                                       ifelse(accuracy.overall > 0.6487, "True Pos", "False Pos"),
#                                       ifelse(accuracy.overall > 0.6487, "False Neg", "True Neg"))]
# dat_1 <- sim.naive.results[!is.na(bias.accuracy), .(percent = .N / 1000,
#                                bias.accuracy = median(bias.accuracy, na.rm = T),
#                                lwr = quantile(bias.accuracy, 0.025, na.rm = T),
#                                upr = quantile(bias.accuracy, 0.975, na.rm = T)),
#                            by = c("k", "target.k.alpha", "n.units_f", "results")]

# pdf("BAYES_summary_04.pdf", width = 12, height = 10, paper = "a4r")
# ggplot(dat_1[results %in% c("False Pos", "False Neg"), ],
#        aes(y = percent, x = factor(target.k.alpha), fill = factor(results))) +
#   geom_bar(stat = "identity") +  facet_grid(k ~ n.units_f) +
#   xlab("Target Kripp alpha values") + ylab("% Decision Error Based on Accuracy (validation vs. true value)") +
#   ggtitle("% Decision error, Naive Bayes") + theme(plot.title = element_text(hjust = 0.5)) +
#   theme(legend.position="bottom") + guides(fill = guide_legend(title = "Error types"))
#
# ggplot(dat_1[results %in% c("False Pos", "False Neg") & percent > 0.01, ],
#        aes(y = bias.accuracy, x = factor(k), color = factor(target.k.alpha))) +
#   geom_point(position = position_dodge(0.7)) +
#   geom_errorbar(aes(ymin = lwr, ymax = upr), position = position_dodge(0.7)) +
#   geom_hline(yintercept = 1, color = "grey", linetype = 2) +
#   facet_grid( ~ n.units_f) +
#   xlab("k = No. of coders") + ylab("False Negative (below 1) vs. False Positive (above 1)") +
#   theme(legend.position="bottom") +
#   ggtitle("Relative Bias in Accuracy (validation vs. true value), Among False Results") +
#   theme(plot.title = element_text(hjust = 0.5)) +
#   guides(color = guide_legend(title = "Target Kripp alpha values"))
# dev.off()
#

sim.naive.results[, results2 := ifelse(Valdat.f > 0.6429,
                                       ifelse(f.overall > 0.6429, "True Pos", "False Pos"),
                                       ifelse(f.overall > 0.6429, "False Neg", "True Neg"))]
dat_2 <- sim.naive.results[!is.na(bias.F1), .(percent = .N / 1000,
                               bias.F1 = median(bias.F1, na.rm = T),
                               lwr = quantile(bias.F1, 0.025, na.rm = T),
                               upr = quantile(bias.F1, 0.975, na.rm = T)),
                           by = c("k", "target.k.alpha", "n.units_f", "results2")]

pdf("BAYES_summary_05.pdf", width = 12, height = 10, paper = "a4r")
ggplot(dat_2[results2 %in% c("False Pos", "False Neg"), ],
       aes(y = percent, x = factor(target.k.alpha), fill = factor(results2))) +
  geom_bar(stat = "identity") +  facet_grid(k ~ n.units_f) +
  xlab("Target Kripp alpha values") + ylab("% Decision Error Based on F1 (validation vs. true value)") +
  ggtitle("% Decision error, Naive Bayes") + theme(plot.title = element_text(hjust = 0.5)) +
  theme(legend.position="bottom") + guides(fill = guide_legend(title = "Error types"))

p4_1 + ggtitle("Relative Bias in F1 (validation vs. true value)") +
  theme(plot.title = element_text(hjust = 0.5))
dev.off()



# ------------------ GLM -------------- #

## ---------------- ##
## Basic desciptive ##
## ---------------- ##

if (!("sim.binomial.results" %in% ls())) load("sim.binomial.results.Rdata")
sim.binomial.results[, n.units_f := factor(n.units, levels = c(50, 100, 250, 500),
                                        labels = c("Annotation N = 50", "Annotation N = 100",
                                                   "Annotation N = 250", "Annotation N = 500"))]
sim.binomial.results[, k_f := factor(k, levels = c(2, 4, 7, 10),
                                  labels = c("Coder \nn = 2", "Coder \nn = 4",
                                             "Coder \nn = 7", "Coder \nn = 10"))]
## overall accuracy as a fuction of target.k.alpha
plot00 <- sim.binomial.results[, .(accuracy = median(accuracy.overall, na.rm = T),
                               lwr = quantile(accuracy.overall, 0.025, na.rm = T),
                               upr = quantile(accuracy.overall, 0.975, na.rm = T)),
                           by = c("k_f", "target.k.alpha", "n.units_f")]

p00 <- ggplot(plot00, aes(x = accuracy, y = target.k.alpha, xmin = lwr, xmax = upr)) +
  geom_point(size = 1.5) + geom_errorbarh(height = 0) +
  xlab("") + ylab("Target K alpha values") +
  ggtitle("Overall Accuracy: GLM") +
  geom_vline(xintercept = sim.naive.results[, mean(accuracy.overall)],
             color = "gray", linetype = 2) + ## reference line is overall mean
  theme_bw() + theme(legend.position="none", plot.title = element_text(hjust = 0.5)) +
  facet_grid(k_f ~ n.units_f)


p5 <- ggplot(sim.binomial.results, aes(x = target.k.alpha, y = accuracy.overall)) +
  geom_smooth(method = "lm", alpha = 0.2, color = "black") + theme_bw() +
  facet_grid( ~ n.units_f) +
  xlab("Target Kripp alpha values") + ylab("Overall Accuracy (against true value)") +
  theme(legend.position="bottom")

p6 <- ggplot(sim.binomial.results, aes(x = target.k.alpha, y = f.overall)) +
  geom_smooth(method = "lm", alpha = 0.2, color = "black") + theme_bw() +
  facet_grid( ~ n.units_f) +
  xlab("Target Kripp alpha values") + ylab("Overall F1 score (using true value)") +
  theme(legend.position="bottom")

pdf("GLM_summary_01.pdf", width = 12, height = 10, paper = "a4r")
p5 + ggtitle("Overall Classification Quality: GLM") +
  theme(plot.title = element_text(hjust = 0.5)) +
  p6 + plot_layout(nrow = 2)
dev.off()

## relative bias
sim.binomial.results[, abs.bias.accuracy := abs((Valdat.accuracy/accuracy.overall) - 1)]
sim.binomial.results[, abs.bias.F1 := abs((Valdat.f/f.overall) - 1)]

# p7 <- ggplot(sim.binomial.results,
#              aes(x = target.k.alpha, y = abs.bias.accuracy, color = factor(k))) +
#   geom_smooth(method = "lm", alpha = 0.2, aes(color = factor(k))) + theme_bw() +
#   facet_grid( ~ n.units_f) +
#   xlab("Target Kripp alpha values") + ylab("Abs Bias of Accuracy (validation vs. true value)") +
#   theme(legend.position="none") +
#   guides(color = guide_legend(title = "No of coders"))
#
# p8 <- ggplot(sim.binomial.results,
#              aes(x = target.k.alpha, y = abs.bias.F1, color = factor(k))) +
#   geom_smooth(method = "lm", alpha = 0.2, aes(color = factor(k))) + theme_bw() +
#   facet_grid( ~ n.units_f) +
#   xlab("Target Kripp alpha values") + ylab("Abs Bias of F1 (validation vs. true value)") +
#   theme(legend.position="bottom") +
#   guides(color = guide_legend(title = "No of coders"))
#
# pdf("GLM_summary_02.pdf", width = 12, height = 10, paper = "a4r")
# p7 + ggtitle("Absolute Degree of Bias Against True Values: GLM") +
#   theme(plot.title = element_text(hjust = 0.5)) + p8 + plot_layout(nrow = 2)
# dev.off()
#

## alternatively,
sim.binomial.results[, bias.accuracy := (Valdat.accuracy/accuracy.overall)]
sim.binomial.results[, bias.F1 := (Valdat.f/f.overall)]

p7_1 <- sim.binomial.results[, .(bias.accuracy = median(bias.accuracy, na.rm = T),
                              lwr = quantile(bias.accuracy, 0.025, na.rm = T),
                              upr = quantile(bias.accuracy, 0.975, na.rm = T)),
                          by = c("k", "target.k.alpha", "n.units_f")] %>%
  ggplot(., aes(y = bias.accuracy, x = factor(k), color = factor(target.k.alpha))) +
  geom_point(position = position_dodge(0.7)) +
  geom_errorbar(aes(ymin = lwr, ymax = upr), position = position_dodge(0.7)) +
  geom_hline(yintercept = 1, color = "grey", linetype = 2) +
  facet_grid( ~ n.units_f) +
  xlab("k = No. of coders") + ylab("Relative Bias in Accuracy (validation vs. true value)") +
  theme(legend.position="none") +
  guides(color = guide_legend(title = "Target Kripp alpha values"))

p8_1 <- sim.binomial.results[, .(bias.F1 = median(bias.F1, na.rm = T),
                              lwr = quantile(bias.F1, 0.025, na.rm = T),
                              upr = quantile(bias.F1, 0.975, na.rm = T)),
                          by = c("k", "target.k.alpha", "n.units_f")] %>%
  ggplot(., aes(y = bias.F1, x = factor(k), color = factor(target.k.alpha))) +
  geom_point(position = position_dodge(0.7)) +
  geom_errorbar(aes(ymin = lwr, ymax = upr), position = position_dodge(0.7)) +
  geom_hline(yintercept = 1, color = "grey", linetype = 2) +
  facet_grid( ~ n.units_f) +
  xlab("k = No. of coders") + ylab("Relative Bias in F1 (validation vs. true value)") +
  theme(legend.position="bottom") +
  guides(color = guide_legend(title = "Target Kripp alpha values"))

# pdf("GLM_summary_03.pdf", width = 12, height = 10, paper = "a4r")
# p7_1 + ggtitle("Relative Bias Against True Values: GLM") +
#   theme(plot.title = element_text(hjust = 0.5)) + p8_1 + plot_layout(nrow = 2)
# dev.off()


## ---------------------------------------------------------- ##
## Amaong those studies which pass the validation check!      ##
## cutoff value is the mean of accuracy/F1 score from Study 1 ##
## ---------------------------------------------------------- ##

sim.binomial.results[, results := ifelse(Valdat.accuracy > 0.6487,
                                         ifelse(accuracy.overall > 0.6487, "True Pos", "False Pos"),
                                         ifelse(accuracy.overall > 0.6487, "False Neg", "True Neg"))]
dat_3 <- sim.binomial.results[!is.na(bias.accuracy), .(percent = .N / 1000,
                                  bias.accuracy = median(bias.accuracy, na.rm = T),
                                  lwr = quantile(bias.accuracy, 0.025, na.rm = T),
                                  upr = quantile(bias.accuracy, 0.975, na.rm = T)),
                              by = c("k", "target.k.alpha", "n.units_f", "results")]

# pdf("GLM_summary_04.pdf", width = 12, height = 10, paper = "a4r")
# ggplot(dat_3[results %in% c("False Pos", "False Neg"), ],
#        aes(y = percent, x = factor(target.k.alpha), fill = factor(results))) +
#   geom_bar(stat = "identity") +  facet_grid(k ~ n.units_f) +
#   xlab("Target Kripp alpha values") + ylab("% Decision Error Based on Accuracy (validation vs. true value)") +
#   ggtitle("% Decision error, GLM") + theme(plot.title = element_text(hjust = 0.5)) +
#   theme(legend.position="bottom") + guides(fill = guide_legend(title = "Error types"))
#
# ggplot(dat_3[results %in% c("False Pos", "False Neg") & percent > 0.01, ],
#        aes(y = bias.accuracy, x = factor(k), color = factor(target.k.alpha))) +
#   geom_point(position = position_dodge(0.7)) +
#   geom_errorbar(aes(ymin = lwr, ymax = upr), position = position_dodge(0.7)) +
#   geom_hline(yintercept = 1, color = "grey", linetype = 2) +
#   facet_grid( ~ n.units_f) +
#   xlab("k = No. of coders") + ylab("False Negative (below 1) vs. False Positive (above 1)") +
#   theme(legend.position="bottom") +
#   ggtitle("Relative Bias in Accuracy (validation vs. true value), Among False Results") +
#   theme(plot.title = element_text(hjust = 0.5)) +
#   guides(color = guide_legend(title = "Target Kripp alpha values"))
# dev.off()


sim.binomial.results[, results2 := ifelse(Valdat.f > 0.6429,
                                         ifelse(f.overall > 0.6429, "True Pos", "False Pos"),
                                         ifelse(f.overall > 0.6429, "False Neg", "True Neg"))]
dat_4 <- sim.binomial.results[!is.na(bias.F1), .(percent = .N / 1000,
                                  bias.F1 = median(bias.F1, na.rm = T),
                                  lwr = quantile(bias.F1, 0.025, na.rm = T),
                                  upr = quantile(bias.F1, 0.975, na.rm = T)),
                              by = c("k", "target.k.alpha", "n.units_f", "results2")]

pdf("GLM_summary_05.pdf", width = 12, height = 10, paper = "a4r")
ggplot(dat_4[results2 %in% c("False Pos", "False Neg"), ],
       aes(y = percent, x = factor(target.k.alpha), fill = factor(results2))) +
  geom_bar(stat = "identity") +  facet_grid(k ~ n.units_f) +
  xlab("Target Kripp alpha values") + ylab("% Decision Error Based on F1 (validation vs. true value)") +
  ggtitle("% Decision error, GLM") + theme(plot.title = element_text(hjust = 0.5)) +
  theme(legend.position="bottom") + guides(fill = guide_legend(title = "Error types"))

p8_1 + ggtitle("Relative Bias in F1 (validation vs. true value)") +
  theme(plot.title = element_text(hjust = 0.5))
dev.off()


# ------------- Bag of Words ----------------

if (!("sim.bow.results" %in% ls())) load("sim.bow.results.Rdata")
sim.bow.results[, n.units_f := factor(n.units, levels = c(50, 100, 250, 500),
                                      labels = c("Annotation N = 50", "Annotation N = 100",
                                                 "Annotation N = 250", "Annotation N = 500"))]
sim.bow.results[, k_f := factor(k, levels = c(2, 4, 7, 10),
                                labels = c("Coder \nn = 2", "Coder \nn = 4",
                                           "Coder \nn = 7", "Coder \nn = 10"))]
## overall accuracy as a fuction of target.k.alpha
plot000 <- sim.bow.results[, .(accuracy = median(accuracy.overall, na.rm = T),
                                   lwr = quantile(accuracy.overall, 0.025, na.rm = T),
                                   upr = quantile(accuracy.overall, 0.975, na.rm = T)),
                               by = c("k_f", "target.k.alpha", "n.units_f")]

p000 <- ggplot(plot000, aes(x = accuracy, y = target.k.alpha, xmin = lwr, xmax = upr)) +
  geom_point(size = 1.5) + geom_errorbarh(height = 0) +
  xlab("") + ylab("Target K alpha values") +
  ggtitle("Overall Accuracy: Bag of Words") +
  geom_vline(xintercept = sim.naive.results[, mean(accuracy.overall)],
             color = "gray", linetype = 2) + ## reference line is overall mean
  theme_bw() + theme(legend.position="none", plot.title = element_text(hjust = 0.5)) +
  facet_grid(k_f ~ n.units_f)


p9 <- ggplot(sim.bow.results, aes(x = target.k.alpha, y = accuracy.overall)) +
  geom_smooth(method = "lm", alpha = 0.2, color = "black") + theme_bw() +
  facet_grid( ~ n.units_f) +
  xlab("Target Kripp alpha values") + ylab("Overall Accuracy (against true value)") +
  theme(legend.position="bottom")

p10 <- ggplot(sim.bow.results, aes(x = target.k.alpha, y = f.overall)) +
  geom_smooth(method = "lm", alpha = 0.2, color = "black") + theme_bw() +
  facet_grid( ~ n.units_f) +
  xlab("Target Kripp alpha values") + ylab("Overall F1 score (using true value)") +
  theme(legend.position="bottom")

pdf("BoW_summary_01.pdf", width = 12, height = 10, paper = "a4r")
p9 + ggtitle("Overall Classification Quality: Bag of Words") +
  theme(plot.title = element_text(hjust = 0.5)) +
  p10 + plot_layout(nrow = 2)
dev.off()

## relative bias
sim.bow.results[, abs.bias.accuracy := abs((Valdat.accuracy/accuracy.overall) - 1)]
sim.bow.results[, abs.bias.F1 := abs((Valdat.f/f.overall) - 1)]

# p11 <- ggplot(sim.bow.results,
#               aes(x = target.k.alpha, y = abs.bias.accuracy, color = factor(k))) +
#   geom_smooth(method = "lm", alpha = 0.2, aes(color = factor(k))) + theme_bw() +
#   facet_grid( ~ n.units_f) +
#   xlab("Target Kripp alpha values") + ylab("Abs Bias of Accuracy (validation vs. true value)") +
#   theme(legend.position="none") +
#   guides(color = guide_legend(title = "No of coders"))
#
# p12 <- ggplot(sim.bow.results,
#               aes(x = target.k.alpha, y = abs.bias.F1, color = factor(k))) +
#   geom_smooth(method = "lm", alpha = 0.2, aes(color = factor(k))) + theme_bw() +
#   facet_grid( ~ n.units_f) +
#   xlab("Target Kripp alpha values") + ylab("Abs Bias of F1 (validation vs. true value)") +
#   theme(legend.position="bottom") +
#   guides(color = guide_legend(title = "No of coders"))
#
# pdf("BoW_summary_02.pdf", width = 12, height = 10, paper = "a4r")
# p11 + ggtitle("Absolute Degree of Bias Against True Values: Bag of Words") +
#   theme(plot.title = element_text(hjust = 0.5)) + p12 + plot_layout(nrow = 2)
# dev.off()
#

## alternatively,
sim.bow.results[, bias.accuracy := (Valdat.accuracy/accuracy.overall)]
sim.bow.results[, bias.F1 := (Valdat.f/f.overall)]

p11_1 <- sim.bow.results[, .(bias.accuracy = median(bias.accuracy, na.rm = T),
                             lwr = quantile(bias.accuracy, 0.025, na.rm = T),
                             upr = quantile(bias.accuracy, 0.975, na.rm = T)),
                         by = c("k", "target.k.alpha", "n.units_f")] %>%
  ggplot(., aes(y = bias.accuracy, x = factor(k), color = factor(target.k.alpha))) +
  geom_point(position = position_dodge(0.7)) +
  geom_errorbar(aes(ymin = lwr, ymax = upr), position = position_dodge(0.7)) +
  geom_hline(yintercept = 1, color = "grey", linetype = 2) +
  facet_grid( ~ n.units_f) +
  xlab("k = No. of coders") + ylab("Relative Bias in Accuracy (validation vs. true value)") +
  theme(legend.position="none") +
  guides(color = guide_legend(title = "Target Kripp alpha values"))

p12_1 <- sim.bow.results[, .(bias.F1 = median(bias.F1, na.rm = T),
                             lwr = quantile(bias.F1, 0.025, na.rm = T),
                             upr = quantile(bias.F1, 0.975, na.rm = T)),
                         by = c("k", "target.k.alpha", "n.units_f")] %>%
  ggplot(., aes(y = bias.F1, x = factor(k), color = factor(target.k.alpha))) +
  geom_point(position = position_dodge(0.7)) +
  geom_errorbar(aes(ymin = lwr, ymax = upr), position = position_dodge(0.7)) +
  geom_hline(yintercept = 1, color = "grey", linetype = 2) +
  facet_grid( ~ n.units_f) +
  xlab("k = No. of coders") + ylab("Relative Bias in F1 (validation vs. true value)") +
  theme(legend.position="bottom") +
  guides(color = guide_legend(title = "Target Kripp alpha values"))

# pdf("BoW_summary_03.pdf", width = 12, height = 10, paper = "a4r")
# p11_1 + ggtitle("Relative Bias Against True Values: Bag of Words") +
#   theme(plot.title = element_text(hjust = 0.5)) + p12_1 + plot_layout(nrow = 2)
# dev.off()


## ---------------------------------------------------------- ##
## Amaong those studies which pass the validation check!      ##
## cutoff value is the mean of accuracy/F1 score from Study 1 ##
## ---------------------------------------------------------- ##

sim.bow.results[, results := ifelse(Valdat.accuracy > 0.6487,
                                    ifelse(accuracy.overall > 0.6487, "True Pos", "False Pos"),
                                    ifelse(accuracy.overall > 0.6487, "False Neg", "True Neg"))]
dat_5 <- sim.bow.results[!is.na(bias.accuracy), .(percent = .N / 1000,
                             bias.accuracy = median(bias.accuracy, na.rm = T),
                             lwr = quantile(bias.accuracy, 0.025, na.rm = T),
                             upr = quantile(bias.accuracy, 0.975, na.rm = T)),
                         by = c("k", "target.k.alpha", "n.units_f", "results")]

# pdf("BoW_summary_04.pdf", width = 12, height = 10, paper = "a4r")
# ggplot(dat_5[results %in% c("False Pos", "False Neg"), ],
#        aes(y = percent, x = factor(target.k.alpha), fill = factor(results))) +
#   geom_bar(stat = "identity") +  facet_grid(k ~ n.units_f) +
#   xlab("Target Kripp alpha values") + ylab("% Decision Error Based on Accuracy (validation vs. true value)") +
#   ggtitle("% Decision error, Bag of Words") + theme(plot.title = element_text(hjust = 0.5)) +
#   theme(legend.position="bottom") + guides(fill = guide_legend(title = "Error types"))
#
# ggplot(dat_5[results %in% c("False Pos", "False Neg") & percent > 0.01, ],
#        aes(y = bias.accuracy, x = factor(k), color = factor(target.k.alpha))) +
#   geom_point(position = position_dodge(0.7)) +
#   geom_errorbar(aes(ymin = lwr, ymax = upr), position = position_dodge(0.7)) +
#   geom_hline(yintercept = 1, color = "grey", linetype = 2) +
#   facet_grid( ~ n.units_f) +
#   xlab("k = No. of coders") + ylab("False Negative (below 1) vs. False Positive (above 1)") +
#   theme(legend.position="bottom") +
#   ggtitle("Relative Bias in Accuracy (validation vs. true value), Among False Results") +
#   theme(plot.title = element_text(hjust = 0.5)) +
#   guides(color = guide_legend(title = "Target Kripp alpha values"))
# dev.off()


sim.bow.results[, results2 := ifelse(Valdat.f > 0.6429,
                                     ifelse(f.overall > 0.6429, "True Pos", "False Pos"),
                                     ifelse(f.overall > 0.6429, "False Neg", "True Neg"))]
dat_6 <- sim.bow.results[!is.na(bias.F1), .(percent = .N / 1000,
                             bias.F1 = median(bias.F1, na.rm = T),
                             lwr = quantile(bias.F1, 0.025, na.rm = T),
                             upr = quantile(bias.F1, 0.975, na.rm = T)),
                         by = c("k", "target.k.alpha", "n.units_f", "results2")]

pdf("BoW_summary_05.pdf", width = 12, height = 10, paper = "a4r")
ggplot(dat_6[results2 %in% c("False Pos", "False Neg"), ],
       aes(y = percent, x = factor(target.k.alpha), fill = factor(results2))) +
  geom_bar(stat = "identity") +  facet_grid(k ~ n.units_f) +
  xlab("Target Kripp alpha values") + ylab("% Decision Error Based on F1 (validation vs. true value)") +
  ggtitle("% Decision error, Bag of Words") + theme(plot.title = element_text(hjust = 0.5)) +
  theme(legend.position="bottom") + guides(fill = guide_legend(title = "Error types"))

p12_1 + ggtitle("Relative Bias in F1 (validation vs. true value)") +
  theme(plot.title = element_text(hjust = 0.5))
dev.off()


## overall classification accuracy
require(patchwork)
test <- rbind(plot0, plot00, plot000)
test[, classifier := factor(rep(c("NB", "GLM", "BoW"), each = 80), levels = c("NB", "GLM", "BoW"))]
pdf("overall_accuracy.pdf", height = 18, width = 7, paper = "a4")
ggplot(test, aes(x = accuracy, y = factor(target.k.alpha), xmin = lwr, xmax = upr)) +
  geom_point(size = 1) + geom_errorbarh(height = 0) +
  xlab("Overall Accuracy") + ylab("Target K alpha values") +
  geom_vline(xintercept = sim.naive.results[, mean(accuracy.overall)],
             color = "gray", linetype = 2) + ## reference line is overall mean
  theme_bw() + theme(legend.position="none", plot.title = element_text(hjust = 0.5)) +
  facet_grid(classifier + k_f ~ n.units_f)
dev.off()
