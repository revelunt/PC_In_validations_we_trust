

## reliability testing
list.of.packages <- c("data.table", "tidyverse", "dplyr", "readxl", "irr",
                      "ggplot2", "rstudioapi", "stringi", "tidyr")
new.packages <- list.of.packages[!(list.of.packages %in%
                                     installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, dependencies = T)

## automatically setting working directories
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(ggplot2)
library(tidyr)
library(dplyr)
library(data.table)
library(haven)
library(magrittr)
theme_set(theme_bw())


## read-in the data
reliability.data <- readxl::read_xlsx("Dat/Reliability_CCR_public.xlsx")
setDT(reliability.data)
reliability.data <- reliability.data[-c(1:3), 2:24]
colnames(reliability.data) <- c("ID", "Author", "Year", "Journal",
                               "Coder", "NoFullText", "Relevant",
                               "MethodsUsed", "N_Total", "GoldStandard",
                               "N_ManualCoders", "N_ManuallyCodedTextEntries",
                               "INTRAcoderReliability", "INTERcoderReliability",
                               "ReliabilityType", "ReliabilityValue", "ExactValuesReported",
                               "ValidationProcedures", "Recall_Sensitivity", "Precision",
                               "F_measure", "Other", "Comments")

library(tidyr)
require(irr)

## relevance of an article (alpha = 1)
spread(reliability.data[, Relevant, by = c("ID", "Coder")],
       key = "ID", value = "Relevant")[, -1] %>%
  as.matrix(.) %>% kripp.alpha(., method = "nominal")

# "Automated Method Used" (alpha = 1)
spread(reliability.data[Relevant == 1, MethodsUsed, by = c("ID", "Coder")],
       key = "ID", value = "MethodsUsed")[, -1] %>%
  as.matrix(.) %>% kripp.alpha(., method = "nominal")

# "Refer to Gold Standard?" (alpha = 1)
spread(reliability.data[Relevant == 1, GoldStandard, by = c("ID", "Coder")],
       key = "ID", value = "GoldStandard")[, -1] %>%
  as.matrix(.) %>% kripp.alpha(., method = "nominal")

# "Report intercoder reliability?" (alpha = 1)
spread(reliability.data[Relevant == 1,
                        INTERcoderReliability, by = c("ID", "Coder")],
       key = "ID", value = "INTERcoderReliability")[, -1] %>%
  as.matrix(.) %>% kripp.alpha(., method = "nominal")

# Refer to validation / Report validation measures?
test <- merge(reliability.data[Relevant == 1,
                 car::recode(Recall_Sensitivity,
                             "NA = NA; 'NA' = NA; else = 1"),
                 by = c("ID", "Coder")],
      reliability.data[Relevant == 1,
                 car::recode(Precision,
                             "NA = NA; 'NA' = NA; else = 1"),
                 by = c("ID", "Coder")],
      by = c("ID", "Coder"))

test[, measures := rowSums(.SD, na.rm = T),
       .SDcols = c("V1.x", "V1.y")]
test[, measures := car::recode(measures, "2=1")]

spread(test[, measures, by = c("ID", "Coder")],
       key = "ID", value = "measures")[, -1] %>%
  as.matrix(.) %>% kripp.alpha(., method = "nominal")
