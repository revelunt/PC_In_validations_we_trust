

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
reliability.data <- readxl::read_xlsx("Reliability_CCR.xlsx")
setDT(reliability.data)

study1data <- readxl::read_xlsx("When does garbage stink_ ArticleCodingFile.xlsx")
reliability.data <- reliability.data[-c(1:4), 2:24]
colnames(reliability.data) <- c("ID", "Author", "Year", "Journal",
                               "Coder", "NoFullText", "Relevant",
                               "MethodsUsed", "N_Total", "GoldStandard",
                               "N_ManualCoders", "N_ManuallyCodedTextEntries",
                               "INTRAcoderReliability", "INTERcoderReliability",
                               "ReliabilityType", "ReliabilityValue", "ExactValuesReported",
                               "ValidationProcedures", "Recall_Sensitivity", "Precision",
                               "F_measure", "Other", "Comments")

library(tidyr)

## relevance of an article (alpha = 1)
spread(reliability.data[, Relevant, by = c("ID", "Coder")],
       key = "ID", value = "Relevant")[, -1] %>%
  as.matrix(.) %>% kripp.alpha(., method = "nominal")

# "Automated Method Used" (alpha = 1)
spread(reliability.data[, MethodsUsed, by = c("ID", "Coder")],
       key = "ID", value = "MethodsUsed")[, -1] %>%
  as.matrix(.) %>% kripp.alpha(., method = "nominal")

# "Refer to Gold Standard?" (alpha = 1)
spread(reliability.data[, GoldStandard, by = c("ID", "Coder")],
       key = "ID", value = "GoldStandard")[, -1] %>%
  as.matrix(.) %>% kripp.alpha(., method = "nominal")

# "Report intercoder reliability?" (alpha = 1)
spread(reliability.data[, INTERcoderReliability, by = c("ID", "Coder")],
       key = "ID", value = "INTERcoderReliability")[, -1] %>%
  as.matrix(.) %>% kripp.alpha(., method = "nominal")

# "Intercoder reliability type reported?" (alpha = 0.6) -- Too few fairs
reliability.data[, ReliabilityType := as.numeric(
  car::recode(ReliabilityType, "1 = 1; 2 = 2; 3 = 3; 4 = 4; else = NA"))]
spread(reliability.data[, ReliabilityType, by = c("ID", "Coder")],
       key = "ID", value = "ReliabilityType")[, -1] %>%
  as.matrix(.) %>% kripp.alpha(., method = "nominal")

# "Intercoder reliability type reported?" (alpha = 0.6) -- Too few fairs
reliability.data[, ReliabilityType := as.numeric(
  car::recode(ValidationProcedures, "0 = 0; 1 = 1; else = NA"))]
spread(reliability.data[, ValidationProcedures, by = c("ID", "Coder")],
       key = "ID", value = "ValidationProcedures")[, -1] %>%
  as.matrix(.) %>% kripp.alpha(., method = "nominal")
