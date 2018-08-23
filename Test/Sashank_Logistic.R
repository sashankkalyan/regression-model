library("caret")
library("e1071")
library("ggplot2")
library("MASS")
library("dplyr")
library("Amelia")

mydata <- read.csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
## view the first few rows of the data
head(mydata)

