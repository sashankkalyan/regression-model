"AUTHOR : Sashank.K (May 2018)"


library("caret")
library("e1071")
library("ggplot2")
library("MASS")
library("DAAG")
library("bootstrap")
library("leaps")
library("ROCR")


"install 'caret' package"

"TEMPLATE for csv file : Column 1 is Y - Dependent Var (Responses)and Col 2,3,4... are X - Independent Var (Predictors) "

mydata <- read.csv(filename <- file.choose(),header = F,skip=1)
head(mydata)

for(i in 1:ncol(mydata))
print(class(mydata[[i]]))

"DATA CLEANING"
mydata <- na.omit(mydata)

my.class <- sapply(mydata, class)
nrow(mydata)
my.class

"Factoring Categorical Variables"
for(i in 1:ncol(mydata))
  if(my.class[i] == "factor")  
  mydata[i] = as.factor(mydata[[i]])

"Partitioning the data frame into two"
train_size <- floor(0.70 * nrow(mydata))

## set the seed to make the partition reproductible
set.seed(1231)

train_set <- sample.int(nrow(mydata), size = train_size)

train <- mydata[train_set,]
test <- mydata[-train_set,]

if(my.class[1] == "numeric") {  "Numeric dependent variables --> Linear Model" 
"running Linear Model on the training data"
relation <- lm(formula = V1 ~ ., data = train)
result <- summary(relation)
result
vif(relation)

AIC(relation)
BIC(relation)

"10-fold Cross-Validation"
train_c <- trainControl(method="cv",number=10)
model <- train(V1 ~ . ,data = train, method="lm", trControl = train_c)
summary(model)

varImp(model)
plot(varImp(model))

"best subset"
best.subset <- regsubsets(train$V1~., train, nvmax=5)
best.subset.summary <- summary(best.subset)
best.subset.summary$outmat
best <- data.frame(best.subset.summary$adjr2,best.subset.summary$cp,best.subset.summary$bic)
best

"Step-Wise Regression for variable/feature selection"
step <- stepAIC(relation, direction= "both")
step$anova # display results

"getting final_model from stepAIC"
selected_var <- formula(step)

"Fitting final_model on train set"
fit_relation <- lm(formula = selected_var, data = train)
fit_result <- summary(fit_relation)
fit_result

#Stepwise Selection with BIC
n = dim(train)[1]
stepBIC = stepAIC(fit_relation,k=log(n))
AIC(stepBIC)
BIC(stepBIC)
vif(stepBIC)

"Predicting Test values based on Training data"
Pred <- predict(fit_relation,test)
t_pred <- data.frame(obs=test$V1,pred=Pred)

"Test data RMSE and R-Squared"
defaultSummary(t_pred)

"Putting the predicted values into a dataframe"
test_fit <- data.frame(cbind(Original = test[[1]],Pred))
test_fit$error <- with(test, Pred-test[[1]])
test_fit

"Calculating TRAIN RMSE and R Squared"
Train.MSE <- mean(result$residuals^2)
Train.RMSE <- sqrt(mean(result$residuals^2))

train_output <- data.frame(row.names="Training Set",Train.MSE,Train.RMSE,result$r.squared,result$adj.r.squared)
colnames(train_output)[3] <- "Train.Rsquared"
colnames(train_output)[4] <- "Train.Adj.Rsquared"

"Calculating TEST RMSE and R Squared - #1"
Test.Rsquared <- postResample(Pred,test[[1]])[2]

"MSE - Formula #1"
test_fit.mse <- with(test_fit, mean(error^2))

"Error = Actual value - Observed value"
Test.MSE <- mean((test$V1 - predict.lm(relation,test))^2)
Test.RMSE = sqrt(Test.MSE)

test_output <- data.frame(row.names = "Test Set",Test.MSE,Test.RMSE,Test.Rsquared)

train_output
test_output

"Summary of Training Model"
summary(relation$model)

"Training Set"
plot(train[[1]] ~ train[[2]],data = train,col="red",xlab="Training X",ylab="Training Y")

"Test Set"
plot(test[[1]] ~ test[[2]],data = test,xlab="Test X",ylab="Test Y",col="black")

"Predicted Values"
qqplot(test_fit$Original,test_fit$Pred,col="green",xlab = "Actual Values",ylab="Predicted Values")
abline(relation,col="red")

} else if(my.class[1]== "factor")  {  "Categorical dependent variables --> Logistic model"
  "Multinominal Logistic"
  log_model <- glm (V1 ~. , data = train, family = binomial(link='logit'))
  summary(log_model)
  vif(log_model)
  anova(log_model,test="chisq")
  confint(log_model)
  
  "Predict"
  predict <- predict(log_model,test, type = 'response')
  
  #confusion matrix
  table(test$V1, predict > 0.5)
  
  pr = prediction(predict, test$V1)
  prf = performance(pr, measure = 'tpr', x.measure = 'fpr')
  plot(prf)
  abline(0,1, lwd = 2, lty = 2)
  
  } else {
  
  }