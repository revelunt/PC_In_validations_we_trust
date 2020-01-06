
list.of.packages <- c("data.table", "tidyverse", "dplyr", "readxl", "irr", "car",
                      "ggplot2", "rstudioapi", "stringi", "tidyr")
new.packages <- list.of.packages[!(list.of.packages %in%
                                     installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, dependencies = T)

## automatically setting working directories
library(ggplot2)
library(tidyr)
library(dplyr)
library(data.table)
library(haven)
library(magrittr)
theme_set(theme_bw())

lit.data <- readxl::read_xlsx("Dat/ArticleCodingFile.xlsx") %>% setDT
lit.data <- lit.data[-c(1:3),-1]

colnames(lit.data) <- c("ID", "author", "year", "journal", "title", "coder", "no_fulltext", "relevance",
                        "method_used", "entities", "goldstandard_reported", "N_manual_coder", "N_entires", "intracoder_R",
                        "intercoder_R", "reliability_1", "reliability_2", "reliability_3", "other_reliability",
                        "average_R", "types_reported",
                        "validation_reported", "recall_senisivity", "precision_PPV",
                        "F_measure", "specificity", "NPV", "accuracy", "predict_accurate_1", "predict_accurate_2",
                        "other", "comments")

## there are few articles employ more than two methods (1 = dictionary, 3 = topic modeling)
lit.data[method_used == '1,3', method_used := '1']
lit.data[, method_used := as.numeric(method_used)]
## convert characters to integers
suppressWarnings(lit.data[, reliability_1 := as.numeric(reliability_1)])
suppressWarnings(lit.data[, N_manual_coder := as.numeric(N_manual_coder)])
suppressWarnings(lit.data[, N_entires := as.numeric(N_entires)])
suppressWarnings(lit.data[, recall_senisivity := as.numeric(recall_senisivity)])
suppressWarnings(lit.data[, precision_PPV := as.numeric(precision_PPV)])
suppressWarnings(lit.data[, F_measure := as.numeric(F_measure)])
suppressWarnings(lit.data[, specificity := as.numeric(specificity)])
suppressWarnings(lit.data[, NPV := as.numeric(NPV)])
suppressWarnings(lit.data[, accuracy := as.numeric(accuracy)])
suppressWarnings(lit.data[, predict_accurate_1 := as.numeric(predict_accurate_1)])
suppressWarnings(lit.data[, predict_accurate_2 := as.numeric(predict_accurate_2)])
lit.data[, other := car::recode(other, "'NA' = NA")]

## ------- ##
## Table 1 ##
## ------- ##

## this scripts generates raw data reported in Table 1 of the ms (excluding % values)
## total N retrieved
lit.data[, .N]

## include & excluded:
lit.data[method_used %in% c('1','2','1,3'), included_in_analysis := 1]
  ## excluded
  lit.data[is.na(included_in_analysis), .N]
  ## included
  lit.data[included_in_analysis == 1, .N]



## breakdown of method used among eligible entries
lit.data[included_in_analysis == 1, c(total = .N,
                                      dictionary = table(method_used)[1],
                                      SML = table(method_used)[2])]

## refer to validation?
lit.data[(included_in_analysis == 1) & (validation_reported == 1),
         c(total = .N, dictionary = table(method_used)[1], SML = table(method_used)[2])]

## refer to human-coded gold standard?
lit.data[(included_in_analysis == 1) & (validation_reported == 1) & (goldstandard_reported == 1),
         c(total = .N, dictionary = table(method_used)[1], SML = table(method_used)[2])]

## report any intercoder reliability?
lit.data[(included_in_analysis == 1) & (validation_reported == 1) &
           (goldstandard_reported == 1) & (intercoder_R == 1),
         c(total = .N, dictionary = table(method_used)[1], SML = table(method_used)[2])]

## report K alpha?
lit.data[(included_in_analysis == 1) & (validation_reported == 1) &
           (goldstandard_reported == 1) & (intercoder_R == 1) & (!is.na(reliability_1)),
         c(total = .N, dictionary = table(method_used)[1], SML = table(method_used)[2])]


## report N of coders?
lit.data[(included_in_analysis == 1) & (validation_reported == 1) &
           (goldstandard_reported == 1) & (!is.na(N_manual_coder)),
         c(total = .N, dictionary = table(method_used)[1], SML = table(method_used)[2])]

## report N of validation data?
lit.data[(included_in_analysis == 1) & (validation_reported == 1) &
           (goldstandard_reported == 1) & (!is.na(N_entires)),
         c(total = .N, dictionary = table(method_used)[1], SML = table(method_used)[2])]

## report ANY validation metric?
 ## calculate rowwise N of NA entries (report no validation metric, based on 9 coded categories)
lit.data[, validation_metric_n_na :=
lit.data[, rowSums(is.na(.SD)), .SDcols = c("recall_senisivity", "precision_PPV",
                            "F_measure", "specificity", "NPV", "accuracy", "predict_accurate_1", "predict_accurate_2",
                            "other")]
]
lit.data[(included_in_analysis == 1) & (validation_reported == 1) &
           (goldstandard_reported == 1) & (validation_metric_n_na < 9),
         c(total = .N, dictionary = table(method_used)[1], SML = table(method_used)[2])]

## report PROPER val matric?
 ## calculate rowwise N of NA entries (report no proper validation metric, based on 6 categories)
lit.data[, proper_validation_metric_n_na :=
           lit.data[, rowSums(is.na(.SD)), .SDcols = c("recall_senisivity", "precision_PPV",
                                                       "F_measure", "specificity", "NPV", "accuracy")]
         ]
lit.data[(included_in_analysis == 1) & (validation_reported == 1) &
           (goldstandard_reported == 1) & (proper_validation_metric_n_na < 6),
         c(total = .N, dictionary = table(method_used)[1], SML = table(method_used)[2])]
