"AUTHOR : Sashank.K (July 2018)"


library("caret")
library("e1071")
library("ggplot2")
library("DAAG")
library("leaps")
library("ROCR")
library("bestglm")
library("dummies")
library("dplyr")
library("Matrix")
library("easypackages")
library("MASS")

reqdPackages <- c("easypackages","caret","e1071","ggplot2","DAAG","leaps","ROCR","bestglm","dummies","dplyr","Matrix","MASS")
libraries(reqdPackages)
packages(reqdPackages)

getwd()

"install 'caret' package"

print(" TEMPLATE for csv file : Column 1 is Y - Dependent Var (Responses)and Col 2,3,4... are X - Independent Var (Predictors) ")

mydata.head <- read.csv(filename <- file.choose(),header = T, stringsAsFactors = FALSE)
head(mydata.head)
heading <- colnames(mydata.head)
for(i in 1:ncol(mydata.head))
  print(class(mydata.head[[i]]))

mydata.var <- data.frame(heading)
names(mydata.var)[names(mydata.var) == 'heading'] <- 'Variables'
mydata.var$Enter_Factor_For_Categorical_Variables <- ''
write.csv(mydata.var,file="variables.csv")

shell.exec("variables.csv")

"Add 'Factor' to Categorical variables"
varTable <- read.csv("variables.csv",stringsAsFactors = FALSE)

######### Checking for factors variables and converting to factors ######### 
for(i in 1:ncol(mydata.head))
  if(colnames(mydata.head[i]) == varTable[i,2])
    if(is.null(varTable[i,3])) { 
      mydata.head[i] = mydata.head[i]
    } else if(varTable$Enter_Factor_For_Categorical_Variables[i] == "Factor" | varTable$Enter_Factor_For_Categorical_Variables[i] == "factor") {
         mydata.head[[i]] = as.factor(mydata.head[[i]])
    }

names(mydata.head) <- paste0('V', 1:ncol(mydata.head))
mydata <- mydata.head
head(mydata)

for(i in 1:ncol(mydata))
  print(class(mydata[[i]]))

######### Handling missing values ######### 

summary(mydata)
na_col <- sapply(mydata, function(x) sum(is.na(x)))
na_count <- sum(na_col)
na_col

"finding the percentage of missing values"
perc.miss <- NA # initialize the variable
percent.col <- NA

for(i in 1:ncol(mydata)) {
      perc.miss[i] <- (na_col[[i]]/nrow(mydata[i]))*100
}

percent.col <- data.frame(names(na_col),perc.miss)
colnames(percent.col)[1] <- "Variables"
colnames(percent.col)[2] <- "Percent of missing"
percent.col

"Removing columns with more than 50% missing values"
i <- 1
for(i in 1:ncol(mydata)) {
      if(percent.col[i,2]>50) {
          mydata[i] <- NULL
          percent.col <- percent.col[-i,]
          rownames(percent.col) <- NULL
      } 
}

######### Replacing missing numerical values with median - sapply ######### 
  mydata[sapply(mydata, is.numeric)] <- lapply(mydata[sapply(mydata, is.numeric)], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))

  "Displaying the complete data - Without missing values"
sapply(mydata, function(x) sum(is.na(x)))
row_col <- c(nrow(mydata),ncol(mydata))
row_col

"Handling OUTLIERS using box-plot"
for(i in 1:ncol(mydata)) {
  if(class(mydata[[i]]) %in% c("integer","numeric")) {
    mydata[[i]][!mydata[[i]] %in% boxplot.stats(mydata[[i]])$out]
    boxplot.stats(mydata[[i]])$out
    boxplot(mydata[[i]])
  }
}

# Creating a copy of the dataframe to replace OUTLIERS
mydata.iqr <- data.frame(cbind(mydata))

######### Outliers Capping - 1.5 * IQR - Inter Quartile Range ==> Q3-Q1 ==> 75%,25% ######### 
  for(i in 1:ncol(mydata)) {
      if(class(mydata[[i]]) %in% c("integer","numeric")) {
          x <- mydata[[i]]
          qnt  <- quantile(x, probs=c(.25, .75), na.rm = T)
          caps <- quantile(x, probs=c(.05, .95), na.rm = T)
          H <- 1.5 * IQR(x, na.rm = T)
          x[x < (qnt[1] - H)] <- caps[1]
          x[x > (qnt[2] + H)] <- caps[2]
          mydata.iqr[[i]] <- x
      }
    }

head(mydata)
head(mydata.iqr)

mydata <- mydata.iqr

# check if the data frames are identical
identical(mydata.iqr,mydata)

row_col <- c(nrow(mydata),ncol(mydata))
row_col

########### Adding dummy variables - #1 Model Matrix ########### 
final.mydata <- model.matrix( ~ .+0 ,data = mydata)
finalmyData <- data.frame(final.mydata)

"#2 Sparse model matrix"
sparse.model.matrix( ~ . , data = mydata)

"#3 Caret package - dummyVars"
dm <- dummyVars("~ . ",data= mydata)
str(dm)
data.frame(predict(dm, newdata = mydata))

which.dummy(mydata)
head(mydata)

######### Partitioning the data frame into two ######### 
train_size <- floor(0.70 * nrow(finalmyData))

## set the seed to make the partition reproductible
set.seed(1231)

train_set <- sample.int(nrow(finalmyData), size = train_size)

train <- finalmyData[train_set,]
test <- finalmyData[-train_set,]

######### All variables ######### 
  "running Linear Model on the training data - Using all variables - Adj R^2"
  relation <- lm(formula = V1 ~ ., data = train)
  result <- summary(relation)
  result$adj.r.squared
  
  names(relation$coefficients)  
  
  "running Linear Model on the training data - Using all variables - AIC"
  all.aic <- AIC(relation)
  
  "running Linear Model on the training data - Using all variables - BIC"
  all.bic <- BIC(relation)
  
  "RMSE"
  all.MSE <- mean(result$residuals^2)
  all.RMSE <- sqrt(mean(result$residuals^2))
  
  "ALL VARIABLE MODEL"
  all.final <- matrix(nrow=3,ncol=1,list(result$adj.r.squared,all.aic,all.bic))
  
  ######### Best subset selection ######### 
  
  "Best subset selection - Largest Adjusted R2 Value"
  best.subset <- regsubsets(train$V1~., train, nvmax=7,really.big=FALSE)
  best.subset.summary <- summary(best.subset)
  
  best.subset.adjr2 <- data.frame(
    Adj.R2 = which.max(best.subset.summary$adjr2),
    CP = which.min(best.subset.summary$cp),
    BIC = which.min(best.subset.summary$bic)
  )
  
          "Extracting the variables of the model - Largest Adj.R2"
          get_best_model_formula <- function(id, object, outcome){
            # get models data
            models <- summary(object)$which[id,-1]
            # Get model predictors
            predictors <- names(which(models == TRUE))
            predictors <- paste(predictors, collapse = "+")
            as.formula(paste0(outcome, "~", predictors))
          }
          
  "Largest adj.R2 value"
      best.model <- get_best_model_formula(best.subset.adjr2$Adj.R2,best.subset,"V1")
  
  "Linear Model with variables giving the largest adj.R2 value"
      best.model_fit <- lm(formula = best.model, data = train)
      best.model.adjr2 <- summary(best.model_fit)
      best.model.RMSE <- sqrt(mean(best.model_fit$residuals^2))
  
  "Smallest BIC value"
      best.model.bic <- get_best_model_formula(best.subset.adjr2$BIC,best.subset,"V1")
  
  "Linear Model with variables giving the smallest BIC value"
      best.model_fit.bic <- lm(formula = best.model, data = train)
      best.model.bic <- summary(best.model_fit)
      best.bic <- BIC(best.model_fit.bic)
      best.bic.RMSE <- sqrt(mean(best.model_fit.bic$residuals^2))
    
  "Linear Model with variables giving the smallest AIC value"
    
        train.glm <- train[,c(2:ncol(train),1)]
        best.model_fit.aic <- bestglm(train.glm,
                            family = gaussian,         # Gaussian family for linear regression
                            IC = "AIC",
                            method = "exhaustive")              # AIC chosen to select models
        summary(res.best$BestModel)
        bestform <- formula(res.best$BestModel)
        bestModel.aic <- lm(formula = bestform,data = train)
        best.aic <- AIC(res.best$BestModel)
  
        "BEST-SUBSET VARIABLE MODEL"
        best.final <- matrix(nrow=3,ncol=1,list(best.model.adjr2$adj.r.squared,best.aic,best.bic))      

  ######### Step-Wise FORWARD Regression ######### 
            
      "Step-Wise FORWARD Regression for variable/feature selection - AIC"
      stepforward.aic <- stepAIC(relation, direction= "forward", trace = FALSE)
      stepforward.aic$anova # display results
      summary(stepforward.aic)
      
              "getting final_model from stepAIC"
              selected_var_for.aic <- formula(stepforward.aic)
              
              "Fitting final_model on train set"
              fit_relation_f_aic <- lm(formula = selected_var_for.aic, data = train)
              fit_result.f <- summary(fit_relation_f_aic)
              fit_result.f$adj.r.squared
              stepf.aic <- AIC(fit_relation_f_aic)
            
      "Step-Wise FORWARD Regression for variable/feature selection - BIC"
      stepBIC.for <- step(relation, direction= "forward", k = log(nrow(train)))
      stepBIC.for$anova # display results
      
              "getting final_model from step BIC"
              selected_var_for <- formula(stepBIC.for)
              
              "Fitting final_model on train set"
              fit_relation_f_bic <- lm(formula = selected_var_for, data = train)
              fit_result <- summary(fit_relation_f_bic)
              stepf.bic <- BIC(fit_relation_f_bic)
              
              "Step FORWARD variable MODEL"
              step.for.final <- matrix(nrow=3,ncol=1,list(fit_result$adj.r.squared,stepf.aic,stepf.bic))
  
  ######### Step-Wise BACKWARD Regression ######### 
        
      "Step-Wise BACKWARD Regression for variable/feature selection - AIC"
      stepbackward.aic <- stepAIC(relation, direction= "backward")
      stepbackward.aic$anova # display results
        
              "getting final_model from stepAIC"
              selected_var_back.aic <- formula(stepbackward.aic)
              
              "Fitting final_model on train set"
              fit_relation_b_aic <- lm(formula = selected_var_back.aic, data = train)
              fit_result_b <- summary(fit_relation_b_aic)
              fit_result_b
              stepb.aic <- AIC(fit_relation_b_aic)
      
      "Step-Wise BACKWARD Regression for variable/feature selection - BIC"
      stepbackward.bic <- step(relation, direction= "backward", k = log(nrow(train)))
      stepbackward.bic$anova # display results
      
             "getting final_model from stepAIC"
              selected_var_back <- formula(stepbackward.bic)
              
              "Fitting final_model on train set"
              fit_relation_b_bic <- lm(formula = selected_var_back, data = train)
              fit_result_b <- summary(fit_relation_b_bic)
              fit_result_b
              stepb.bic <- BIC(fit_relation_b_bic)
        
              "Step BACKWARD variable MODEL"
              step.bk.final <- matrix(nrow=3,ncol=1,list(fit_result_b$adj.r.squared,stepb.aic,stepb.bic))
              
              metric <- matrix(nrow=3,ncol=1,list("Adj.R2","AIC","BIC"))
            
  
  ######### OUTPUT Metric values #########
  
            final_output <- data.frame(cbind(metric,all.final,best.final,step.for.final,step.bk.final))
            
            colnames(final_output)[1] = "Metrics"
            colnames(final_output)[2] = "All variables"
            colnames(final_output)[3] = "Best Subset"
            colnames(final_output)[4] = "Stepwise Forward"
            colnames(final_output)[5] = "Stepwise Backward"
            
       final_output
  
  ######### Predicting TEST values #########
  
  # A list of all relations used for LM function #
  allRelation <- list(relation,best.model_fit,bestModel.aic,best.model_fit.bic,fit_relation_f_aic,fit_relation_f_bic,fit_relation_b_aic,fit_relation_b_bic)
  
  modelName <- list("All variables","Best subset - R2","Best subset - AIC","Best subset - BIC","Stepwise forward - AIC","Stepwise forward - BIC","Stepwise backward - AIC","Stepwise backward - BIC")
  
  predictTest <- NA
  t_pred <- NA
  Test.Rsquared <- NA
  Test.MSE <- NA
  Test.RMSE <- NA
  
  out <- data.frame(Response = test[[1]])
  for(i in 1:length(allRelation)) {
    predictTest[i] <- predict(allRelation[i],test)
    out[,i+1] <- unlist(predictTest[[i]])
    
    Test.Rsquared[i] <- postResample(predictTest[[i]],test[[1]])[2]
    "Error = Actual value - Observed value"
    Test.MSE[i] <- (mean(out[[1]] - out[,i+1])^2)
    Test.RMSE[i] = sqrt(Test.MSE[i])
  }
  
      "Calculating TEST RMSE and R Squared - #1"
      
      metric <- cbind.data.frame(unlist(modelName),Test.MSE,Test.RMSE,Test.Rsquared)
      colnames(metric)[1] <- "Model"
      out
      
      print("Prediction of the test set")
      metric