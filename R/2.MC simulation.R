
## ------------------------------------------------------------------- ##
## Monte Carlo Simulation code for paper: In validations we trust?     ##
## ------------------------------------------------------------------- ##

rm(list = ls())
## install required libraries if not already installed
list.of.packages <- c("data.table", "tidyverse", "dplyr", "arm", "e1071",
                      "ggplot2", "grid", "stringi", "tidyr", "MCMCpack",
                      "irr", "caret", "rstudioapi", "tidytext", "extraDistr",
                      'naivebayes', 'brms', 'rstan')
new.packages <- list.of.packages[!(list.of.packages %in%
                                     installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, dependencies = T)

## automatically setting working directories
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)); setwd('..')
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

n_days = 5*52*5 # timespan = 5 years, 5 days per week (assuming 52 weeks per year)
n_media = 10 # 10 media outlets
n_articles_per_day = 10 # 5 political news per day per outlet

n.total <- n_days*n_media*n_articles_per_day
n.total # total number of media contents to be coded = 260*5*10*5 = 130000

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
## b0 = rnorm(1:n_media), ## fixed effect for media outlets
## b1 = .5, b2 = .2, b3 = .6

## true population parameters
b1 = .5; b2 = .2; b3 = .6

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
#                                       "obs.with.error.3", "obs.with.error.4")] %>% average.agreement()
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

# SML Simulation --------------------------------------------------
## set seed for reproducibility
RNGkind("L'Ecuyer-CMRG")
set.seed(12345)

sim.SML.results <- sim.all.scenario()
save(sim.SML.results, file = "RR1.sim.SML.Rdata")
## took appx. 10h to complete
rm(sim.SML.results); gc()


# Dictionary Simulation -------------------------------------------------
## set seed for reproducibility
RNGkind("L'Ecuyer-CMRG")
set.seed(12345)
sim.dict.results <- sim.all.scenario_dict()
save(sim.dict.results, file = "RR1.sim.dict.Rdata")
## took appx. 3h to complete
rm(sim.dict.results); gc()


## ------------------- ##
## summarizing results ##
## ------------------- ##

if(!("patchwork" %in% installed.packages()[,"Package"]))
  devtools::install_github("thomasp85/patchwork")
require(patchwork)

## ---------------- ##
## Basic desciptive ##
## ---------------- ##

if ("sim.SML.results" %!in% ls()) load("Results/RR1.sim.SML.Rdata")
if ("sim.dict.results" %!in% ls()) load("Results/RR1.sim.dict.Rdata")

data <- sim.SML.results
data2 <- sim.dict.results
data2[, random.sample := ifelse(random.sample == 1, TRUE, FALSE)]
data2[, duplicated.coding := ifelse(duplicated.coding == 1, TRUE, FALSE)]

index <- c(which(is.na(data[, obs.F1])), which(is.na(data[, true.F1]))) %>% unique
# data[index, ] ## those cases are either no true positive or no false negative predicted by algorithm
data[is.na(obs.F1), obs.F1 := 0]
data[is.na(true.F1), true.F1 := 0]

index <- c(which(is.na(data2[, obs.F1])), which(is.na(data2[, true.F1]))) %>% unique


## SML
dat1 <- data[, abs(obs.F1 - true.F1),
      by = .(k, n.units, target.k.alpha, random.sample, duplicated.coding)] %>%
  .[ , .(mean = mean(V1, na.rm = T),
         llci2sd = quantile(V1, 0.025, na.rm = T),
         ulci2sd = quantile(V1, 0.975, na.rm = T),
         llci1sd = quantile(V1, 0.16, na.rm = T),
         ulci1sd = quantile(V1, 0.84, na.rm = T)),
     by = .(n.units, target.k.alpha, k, duplicated.coding, random.sample)]
dat1[, random.sample := rep(c("Random: FALSE", "Random: TRUE"), each = 36, times = 2)]
dat1[, duplicated.coding := rep(c("Duplicated: FALSE", "Duplicated: TRUE"), each = 72)]
dat1[, n.units := ifelse(n.units == 600, "N = 600",
                           ifelse(n.units == 1300, "N = 1300",
                                  ifelse(n.units == 6500, "N = 6500", "N = 13000")))]
dat1[, n.units := factor(n.units, levels = c("N = 600", "N = 1300", "N = 6500", "N = 13000"))]
dat1[, target.k.alpha := ifelse(target.k.alpha == 0.5, "K alpha .5",
                                  ifelse(target.k.alpha == 0.7, "K alpha .7", "K alpha .9"))]

p1a_appndx <- dat1 %>%
  ggplot(., aes(x = factor(k), y = mean, group = factor(target.k.alpha),
                ymin = llci2sd, ymax = ulci2sd,
                color = factor(target.k.alpha))) +
  geom_point(position = position_dodge(width = 0.5)) +
  geom_linerange(position = position_dodge(width = 0.5), alpha = 0.5) +
  geom_linerange(aes(ymin = llci1sd, ymax = ulci1sd),
                 position = position_dodge(width = 0.5),
                 size = 1.5, alpha = 0.7) +
  facet_grid(duplicated.coding ~ random.sample + n.units) +
  ylab("Mean Absolute Prediction Error") + xlab("k = no. of coders") +
  theme(legend.position = "bottom", legend.direction = "horizontal") +
  scale_color_discrete(name = "Krippendorff alpha", labels = c("K alpha = .5", "K alpha = .7", "K alpha = .9"))


dat2 <- data[, abs(obs.F1 - true.F1),
             by = .(k, n.units, target.k.alpha, random.sample, duplicated.coding)] %>%
  .[ , .(mean = mean(V1, na.rm = T),
         llci2sd = quantile(V1, 0.025, na.rm = T),
         ulci2sd = quantile(V1, 0.975, na.rm = T),
         llci1sd = quantile(V1, 0.16, na.rm = T),
         ulci1sd = quantile(V1, 0.84, na.rm = T)),
     by = .(n.units, target.k.alpha, random.sample)]

dat2[, random.sample := rep(c("Random: FALSE", "Random: TRUE"), each = 12)]
dat2[, n.units := ifelse(n.units == 600, "N = 600",
                         ifelse(n.units == 1300, "N = 1300",
                                ifelse(n.units == 6500, "N = 6500", "N = 13000")))]
dat2[, n.units := factor(n.units, levels = c("N = 600", "N = 1300", "N = 6500", "N = 13000"))]
dat2[, target.k.alpha := ifelse(target.k.alpha == 0.5, "K alpha .5",
                                ifelse(target.k.alpha == 0.7, "K alpha .7", "K alpha .9"))]

p1a <-
  ggplot(dat2, aes(x = factor(n.units), y = mean, group = factor(target.k.alpha),
                   ymin = llci2sd, ymax = ulci2sd,
                   color = factor(target.k.alpha))) + ylim(0, 0.4) +
  geom_point(position = position_dodge(width = 0.5)) +
  geom_linerange(position = position_dodge(width = 0.5), alpha = 0.5) +
  geom_linerange(aes(ymin = llci1sd, ymax = ulci1sd),
                 position = position_dodge(width = 0.5),
                 size = 1.5, alpha = 0.7) +
  facet_grid(. ~  random.sample) + ylab("Mean Absolute Prediction Error") + xlab("Size of validation dataset") +
  theme(legend.position = "bottom", legend.direction = "horizontal") +
  scale_color_discrete(name = "Krippendorff alpha", labels = c("K alpha = .5", "K alpha = .7", "K alpha = .9"))

## Quantity reported in the ms
lm.data.SML <-  data[, .(MAPE = mean(abs(obs.F1 - true.F1))),
                      by = .(k, n.units, target.k.alpha, random.sample, duplicated.coding)]

lm.data.SML[, mean(MAPE, na.rm = T), by = duplicated.coding]
lm.data.SML[, mean(MAPE, na.rm = T), by = k]
lm.data.SML[, mean(MAPE, na.rm = T), by = random.sample]
lm.data.SML[, mean(MAPE, na.rm = T), by = target.k.alpha]
lm.data.SML[, mean(MAPE, na.rm = T), by = n.units]

require(car)
lm.SML.main <- lm.data.SML[, aov(
  lm(MAPE ~ factor(k) + factor(n.units) + factor(target.k.alpha) + factor(random.sample) + factor(duplicated.coding)))]
summary(lm.SML.main); TukeyHSD(lm.SML.main)
require(sjstats)
omega_sq(lm.SML.main)

lm.SML.inter <- lm.data.SML[, aov(
  lm(MAPE ~ factor(n.units)*factor(target.k.alpha)*factor(random.sample) + factor(k) + factor(duplicated.coding)))]
summary(lm.SML.inter); TukeyHSD(lm.SML.inter)


## additional interaction checks reported in online appendix
lm.SML.inter1 <- lm.data.SML[, aov(
  lm(MAPE ~ factor(k)*factor(duplicated.coding) + factor(n.units)*factor(duplicated.coding) +
       factor(target.k.alpha)*factor(duplicated.coding) + factor(random.sample)*factor(duplicated.coding)))]
lm.SML.inter2 <- lm.data.SML[, aov(
  lm(MAPE ~ factor(k)*factor(duplicated.coding) + factor(n.units)*factor(k) +
       factor(target.k.alpha)*factor(k) + factor(random.sample)*factor(k)))]


## Dictionary
dat1 <- data2[, abs(obs.F1 - true.F1),
      by = .(k, n.units, target.k.alpha, random.sample, duplicated.coding)] %>%
  .[ , .(mean = mean(V1, na.rm = T),
         llci2sd = quantile(V1, 0.025, na.rm = T),
         ulci2sd = quantile(V1, 0.975, na.rm = T),
         llci1sd = quantile(V1, 0.16, na.rm = T),
         ulci1sd = quantile(V1, 0.84, na.rm = T)),
     by = .(n.units, target.k.alpha, k, duplicated.coding, random.sample)]
dat1[, random.sample := rep(c("Random: TRUE", "Random: FALSE"), each = 36, times = 2)]
dat1[, duplicated.coding := rep(c("Duplicated: TRUE", "Duplicated: FALSE"), each = 72)]
dat1[, n.units := ifelse(n.units == 600, "N = 600",
                         ifelse(n.units == 1300, "N = 1300",
                                ifelse(n.units == 6500, "N = 6500", "N = 13000")))]
dat1[, n.units := factor(n.units, levels = c("N = 600", "N = 1300", "N = 6500", "N = 13000"))]
dat1[, target.k.alpha := ifelse(target.k.alpha == 0.5, "K alpha .5",
                                ifelse(target.k.alpha == 0.7, "K alpha .7", "K alpha .9"))]

p1b_appndx <- dat1 %>% ggplot(., aes(x = factor(k), y = mean, group = factor(target.k.alpha),
                         ymin = llci2sd, ymax = ulci2sd,
                         color = factor(target.k.alpha))) +
  geom_point(position = position_dodge(width = 0.5)) +
  geom_linerange(position = position_dodge(width = 0.5), alpha = 0.5) +
  geom_linerange(aes(ymin = llci1sd, ymax = ulci1sd),
                 position = position_dodge(width = 0.5),
                 size = 1.5, alpha = 0.7) +
  facet_grid(duplicated.coding ~ random.sample + n.units) +
  ylab("Mean Absolute Percentage Error") + xlab("k = no. of coders") +
  theme(legend.position = "bottom", legend.direction = "horizontal") +
  scale_color_discrete(name = "Krippendorff alpha", labels = c("K alpha = .5", "K alpha = .7", "K alpha = .9"))

dat2 <- data2[, abs(obs.F1 - true.F1)/true.F1,
              by = .(k, n.units, target.k.alpha, random.sample, duplicated.coding)] %>%
  .[ , .(mean = mean(V1, na.rm = T),
         llci2sd = quantile(V1, 0.025, na.rm = T),
         ulci2sd = quantile(V1, 0.975, na.rm = T),
         llci1sd = quantile(V1, 0.16, na.rm = T),
         ulci1sd = quantile(V1, 0.84, na.rm = T)),
     by = .(n.units, target.k.alpha, random.sample)]
dat2[, random.sample := rep(c("Random: TRUE", "Random: FALSE"), each = 12)]
dat2[, n.units := ifelse(n.units == 600, "N = 600",
                         ifelse(n.units == 1300, "N = 1300",
                                ifelse(n.units == 6500, "N = 6500", "N = 13000")))]
dat2[, n.units := factor(n.units, levels = c("N = 600", "N = 1300", "N = 6500", "N = 13000"))]
dat2[, target.k.alpha := ifelse(target.k.alpha == 0.5, "K alpha .5",
                                ifelse(target.k.alpha == 0.7, "K alpha .7", "K alpha .9"))]

p1b <-
  ggplot(dat2, aes(x = factor(n.units), y = mean, group = factor(target.k.alpha),
                   ymin = llci2sd, ymax = ulci2sd,
                   color = factor(target.k.alpha))) + ylim(0, 0.4) +
  geom_point(position = position_dodge(width = 0.5)) +
  geom_linerange(position = position_dodge(width = 0.5), alpha = 0.5) +
  geom_linerange(aes(ymin = llci1sd, ymax = ulci1sd),
                 position = position_dodge(width = 0.5),
                 size = 1.5, alpha = 0.7) +
  facet_grid(. ~  random.sample) + ylab("Mean Absolute Prediction Error") + xlab("Size of validation dataset") +
  theme(legend.position = "bottom", legend.direction = "horizontal") +
  scale_color_discrete(name = "Krippendorff alpha", labels = c("K alpha = .5", "K alpha = .7", "K alpha = .9"))


## quantity reported in the ms
lm.data.dict <- data2[, .(MAPE = mean(abs(obs.F1 - true.F1))),
                    by = .(k, n.units, target.k.alpha, random.sample, duplicated.coding)]

lm.data.dict[, mean(MAPE, na.rm = T), by = duplicated.coding]
lm.data.dict[, mean(MAPE, na.rm = T), by = k]
lm.data.dict[, mean(MAPE), by = random.sample]
lm.data.dict[, mean(MAPE, na.rm = T), by = target.k.alpha]
lm.data.dict[, mean(MAPE, na.rm = T), by = n.units]

require(sjstats)
lm.dict.main <- lm.data.dict[, aov(
  lm(MAPE ~ factor(k) + factor(n.units) + factor(target.k.alpha) + factor(random.sample) + factor(duplicated.coding)))]
summary(lm.dict.main); omega_sq(lm.dict.main)

TukeyHSD(lm.dict.main)

lm.dict.inter <- lm.data.dict[, aov(
  lm(MAPE ~ factor(n.units)*factor(target.k.alpha)*factor(random.sample) + factor(k) + factor(duplicated.coding)))]
summary(lm.dict.inter); TukeyHSD(lm.dict.inter)


## duplicated coding and the number of coders do not really matter much.
## proper sampling variability of validation material is most important factor
## under proper sampling variability, both number of total units and
## K alpha decreases bias.
## K alpha compensates the effects of low n. of total units,
## yet alpha effects appear to be more pronounced when n increases as well.


## decision error evaluation, SML
thresholds <- c(0.483, 0.624, 0.766) ## overall mean, +-1SD, SML and dict COMBINED
names(thresholds) <- paste0("F", thresholds)

test <- lapply(thresholds, function(i) {
  classification.table <- data[, ifelse(obs.F1 >= i,
      ifelse(true.F1 >= i, "TP", "FP"), ## if obs.F1 >= i,
      ifelse(true.F1 >= i, "FN", "TN"))] ## if obs.F1 < i
      classification.table
      }) %>% do.call(cbind, .) %>% as.data.frame %>% setDT

data <- cbind(data, test)

total <- 6000

## at F1 = 0.491 (-1SD from mean)
dat1 <- data[F0.483 %in% c("FN", "FP"), .(perc = .N/total*100),
             by = .(n.units, target.k.alpha, random.sample, F0.483)]
dat2 <- expand.grid.df(tibble(n.units = c(600, 1300, 6500, 13000)),
                       tibble(target.k.alpha = c(0.5, 0.7, 0.9)),
                       tibble(random.sample = c(TRUE, FALSE)),
                       tibble(F0.483 = c("FN", "FP"))) %>% setDT(.)
F0.483 <- merge(dat1[order(n.units, target.k.alpha, random.sample),],
        dat2[order(n.units, target.k.alpha, random.sample),],
        by = c("n.units", "target.k.alpha", "random.sample", "F0.483"),
        all = TRUE); rm(dat1, dat2)
F0.483[is.na(perc), perc := 0]
F0.483[, random.sample := rep(c("Random: FALSE", "Random: TRUE"), each = 2, times = 12)]
F0.483[, n.units := ifelse(n.units == 600, "N = 600",
                           ifelse(n.units == 1300, "N = 1300",
                                  ifelse(n.units == 6500, "N = 6500", "N = 13000")))]
F0.483[, n.units := factor(n.units, levels = c("N = 600", "N = 1300", "N = 6500", "N = 13000"))]
F0.483[, target.k.alpha := ifelse(target.k.alpha == 0.5, "K alpha .5",
                           ifelse(target.k.alpha == 0.7, "K alpha .7", "K alpha .9"))]

p2_1 <- ggplot(F0.483, aes(y = perc, x = factor(target.k.alpha), fill = factor(F0.483))) +
  geom_bar(stat = "identity") + facet_grid(random.sample ~ n.units) +
  xlab("") + ylab("% Decision Error \n(validation F1 vs. true F1)") +
  ggtitle("% of Decision error, F1 threshold = .483") + ylim(0, 16) +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(legend.position = "none")


## at F1 = 0.644 (mean)
dat1 <- data[F0.624 %in% c("FN", "FP"), .(perc = .N/total*100),
     by = .(n.units, target.k.alpha, random.sample, F0.624)]
dat2 <- expand.grid.df(tibble(n.units = c(600, 1300, 6500, 13000)),
                       tibble(target.k.alpha = c(0.5, 0.7, 0.9)),
                       tibble(random.sample = c(TRUE, FALSE)),
                       tibble(F0.624 = c("FN", "FP"))) %>% setDT(.)
F0.624 <- merge(dat1[order(n.units, target.k.alpha, random.sample),],
                dat2[order(n.units, target.k.alpha, random.sample),],
                by = c("n.units", "target.k.alpha", "random.sample", "F0.624"),
                all = TRUE); rm(dat1, dat2)
F0.624[is.na(perc), perc := 0]
F0.624[, random.sample := rep(c("Random: FALSE", "Random: TRUE"), each = 2, times = 12)]
F0.624[, n.units := ifelse(n.units == 600, "N = 600",
                           ifelse(n.units == 1300, "N = 1300",
                                  ifelse(n.units == 6500, "N = 6500", "N = 13000")))]
F0.624[, n.units := factor(n.units, levels = c("N = 600", "N = 1300", "N = 6500", "N = 13000"))]
F0.624[, target.k.alpha := ifelse(target.k.alpha == 0.5, "K alpha .5",
                                  ifelse(target.k.alpha == 0.7, "K alpha .7", "K alpha .9"))]

p2_2 <- ggplot(F0.624, aes(y = perc, x = factor(target.k.alpha), fill = factor(F0.624))) +
  geom_bar(stat = "identity") +  facet_grid(random.sample ~ n.units) +
  xlab("") + ylab("% Decision Error \n(validation F1 vs. true F1)") +
  ggtitle("% Decision Error, F1 threshold = 0.624") + ylim(0, 16) +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(legend.position="none")

## at F1 = 0.797 (+1SD from mean)
dat1 <- data[F0.766 %in% c("FN", "FP"), .(perc = .N/total*100),
     by = .(n.units, target.k.alpha, random.sample, F0.766)]
dat2 <- expand.grid.df(tibble(n.units = c(600, 1300, 6500, 13000)),
                       tibble(target.k.alpha = c(0.5, 0.7, 0.9)),
                       tibble(random.sample = c(TRUE, FALSE)),
                       tibble(F0.766 = c("FN", "FP"))) %>% setDT(.)
F0.766 <- merge(dat1[order(n.units, target.k.alpha, random.sample),],
                dat2[order(n.units, target.k.alpha, random.sample),],
                by = c("n.units", "target.k.alpha", "random.sample", "F0.766"),
                all = TRUE); rm(dat1, dat2)
F0.766[is.na(perc), perc := 0]
F0.766[, random.sample := rep(c("Random: FALSE", "Random: TRUE"), each = 2, times = 12)]
F0.766[, n.units := ifelse(n.units == 600, "N = 600",
                           ifelse(n.units == 1300, "N = 1300",
                                  ifelse(n.units == 6500, "N = 6500", "N = 13000")))]
F0.766[, n.units := factor(n.units, levels = c("N = 600", "N = 1300", "N = 6500", "N = 13000"))]
F0.766[, target.k.alpha := ifelse(target.k.alpha == 0.5, "K alpha .5",
                                  ifelse(target.k.alpha == 0.7, "K alpha .7", "K alpha .9"))]

p2_3 <- ggplot(F0.766, aes(y = perc, x = factor(target.k.alpha), fill = factor(F0.766))) +
  geom_bar(stat = "identity") +  facet_grid(random.sample ~ n.units) +
  xlab("") + ylab("% Decision Error \n(validation F1 vs. true F1)") +
  ggtitle("% Decision Error, F1 threshold = 0.766") + ylim(0, 16) +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(legend.position="bottom") +
  scale_fill_discrete(name = "Error types", labels = c("False Negative", "False Positive"))

p2_1 + p2_2 + p2_3 + plot_layout(nrow = 3)

dat <- rbind(F0.483, F0.624, F0.766, use.names=FALSE)
colnames(dat)[4] <- c("error.type")
dat[, F.threshold := rep(thresholds, each = 48)]

# quantity reported in ms.
dat[error.type == "FN", max(perc)]
dat[error.type == "FN", mean(perc), by = target.k.alpha]
dat[error.type == "FP" & F.threshold == 0.766, max(perc)]
dat[error.type == "FP" & F.threshold == 0.766, ]
dat[error.type == "FP", max(perc)]

## decision error evaluation, dict
test2 <- lapply(thresholds, function(i) {
  classification.table <- data2[, ifelse(obs.F1 >= i,
                                        ifelse(true.F1 >= i, "TP", "FP"), ## if obs.F1 >= i,
                                        ifelse(true.F1 >= i, "FN", "TN"))] ## if obs.F1 < i
  classification.table
}) %>% do.call(cbind, .) %>% as.data.frame %>% setDT

data2 <- cbind(data2, test2)

total <- 6000

## at F1 = 0.491 (-1SD from mean)
dat1 <- data2[F0.483 %in% c("FN", "FP"), .(perc = .N/total*100),
             by = .(n.units, target.k.alpha, random.sample, F0.483)]
dat2 <- expand.grid.df(tibble(n.units = c(600, 1300, 6500, 13000)),
                       tibble(target.k.alpha = c(0.5, 0.7, 0.9)),
                       tibble(random.sample = c(TRUE, FALSE)),
                       tibble(F0.483 = c("FN", "FP"))) %>% setDT(.)
F0.483 <- merge(dat1[order(n.units, target.k.alpha, random.sample),],
                dat2[order(n.units, target.k.alpha, random.sample),],
                by = c("n.units", "target.k.alpha", "random.sample", "F0.483"),
                all = TRUE); rm(dat1, dat2)
F0.483[is.na(perc), perc := 0]
F0.483[, random.sample := rep(c("Random: FALSE", "Random: TRUE"), each = 2, times = 12)]
F0.483[, n.units := ifelse(n.units == 600, "N = 600",
                           ifelse(n.units == 1300, "N = 1300",
                                  ifelse(n.units == 6500, "N = 6500", "N = 13000")))]
F0.483[, n.units := factor(n.units, levels = c("N = 600", "N = 1300", "N = 6500", "N = 13000"))]
F0.483[, target.k.alpha := ifelse(target.k.alpha == 0.5, "K alpha .5",
                                  ifelse(target.k.alpha == 0.7, "K alpha .7", "K alpha .9"))]

p3_1 <- ggplot(F0.483, aes(y = perc, x = factor(target.k.alpha), fill = factor(F0.483))) +
  geom_bar(stat = "identity") + facet_grid(random.sample ~ n.units) +
  xlab("") + ylab("% Decision Error (validation F1 vs. true F1)") +
  ggtitle("% of Decision error, F1 threshold = .483") + ylim(0, 16) +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(legend.position = "none")


## at F1 = 0.644 (mean)
dat1 <- data2[F0.624 %in% c("FN", "FP"), .(perc = .N/total*100),
             by = .(n.units, target.k.alpha, random.sample, F0.624)]
dat2 <- expand.grid.df(tibble(n.units = c(600, 1300, 6500, 13000)),
                       tibble(target.k.alpha = c(0.5, 0.7, 0.9)),
                       tibble(random.sample = c(TRUE, FALSE)),
                       tibble(F0.624 = c("FN", "FP"))) %>% setDT(.)
F0.624 <- merge(dat1[order(n.units, target.k.alpha, random.sample),],
                dat2[order(n.units, target.k.alpha, random.sample),],
                by = c("n.units", "target.k.alpha", "random.sample", "F0.624"),
                all = TRUE); rm(dat1, dat2)
F0.624[is.na(perc), perc := 0]
F0.624[, random.sample := rep(c("Random: FALSE", "Random: TRUE"), each = 2, times = 12)]
F0.624[, n.units := ifelse(n.units == 600, "N = 600",
                           ifelse(n.units == 1300, "N = 1300",
                                  ifelse(n.units == 6500, "N = 6500", "N = 13000")))]
F0.624[, n.units := factor(n.units, levels = c("N = 600", "N = 1300", "N = 6500", "N = 13000"))]
F0.624[, target.k.alpha := ifelse(target.k.alpha == 0.5, "K alpha .5",
                                  ifelse(target.k.alpha == 0.7, "K alpha .7", "K alpha .9"))]

p3_2 <- ggplot(F0.624, aes(y = perc, x = factor(target.k.alpha), fill = factor(F0.624))) +
  geom_bar(stat = "identity") +  facet_grid(random.sample ~ n.units) +
  xlab("") + ylab("% Decision Error (validation F1 vs. true F1)") +
  ggtitle("% Decision Error, F1 threshold = 0.624") + ylim(0, 16) +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(legend.position="none")

## at F1 = 0.797 (+1SD from mean)
dat1 <- data2[F0.766 %in% c("FN", "FP"), .(perc = .N/total*100),
             by = .(n.units, target.k.alpha, random.sample, F0.766)]
dat2 <- expand.grid.df(tibble(n.units = c(600, 1300, 6500, 13000)),
                       tibble(target.k.alpha = c(0.5, 0.7, 0.9)),
                       tibble(random.sample = c(TRUE, FALSE)),
                       tibble(F0.766 = c("FN", "FP"))) %>% setDT(.)
F0.766 <- merge(dat1[order(n.units, target.k.alpha, random.sample),],
                dat2[order(n.units, target.k.alpha, random.sample),],
                by = c("n.units", "target.k.alpha", "random.sample", "F0.766"),
                all = TRUE); rm(dat1, dat2)
F0.766[is.na(perc), perc := 0]
F0.766[, random.sample := rep(c("Random: FALSE", "Random: TRUE"), each = 2, times = 12)]
F0.766[, n.units := ifelse(n.units == 600, "N = 600",
                           ifelse(n.units == 1300, "N = 1300",
                                  ifelse(n.units == 6500, "N = 6500", "N = 13000")))]
F0.766[, n.units := factor(n.units, levels = c("N = 600", "N = 1300", "N = 6500", "N = 13000"))]
F0.766[, target.k.alpha := ifelse(target.k.alpha == 0.5, "K alpha .5",
                                  ifelse(target.k.alpha == 0.7, "K alpha .7", "K alpha .9"))]

p3_3 <- ggplot(F0.766, aes(y = perc, x = factor(target.k.alpha), fill = factor(F0.766))) +
  geom_bar(stat = "identity") +  facet_grid(random.sample ~ n.units) +
  xlab("") + ylab("% Decision Error (validation F1 vs. true F1)") +
  ggtitle("% Decision Error, F1 threshold = 0.766") + ylim(0, 16) +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(legend.position="bottom") +
  scale_fill_discrete(name = "Error types", labels = c("False Negative", "False Positive"))

p3_1 + p3_2 + p3_3 + plot_layout(nrow = 3)

dat <- rbind(F0.483, F0.624, F0.766, use.names=FALSE)
colnames(dat)[4] <- c("error.type")
dat[, F.threshold := rep(thresholds, each = 48)]

# quantity reported in ms.
dat[error.type == "FN", max(perc)]
dat[error.type == "FN", mean(perc), by = target.k.alpha]
dat[error.type == "FP" & F.threshold == 0.766, max(perc)]
dat[error.type == "FP" & F.threshold == 0.766, ]
