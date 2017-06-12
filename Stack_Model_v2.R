library('caret')
library('pROC')
data<-read.csv(file.choose(),header = TRUE)
set.seed(1234)
sample_data <- createDataPartition(y = data$y, p = .80, list = FALSE)
trainingset <- data[ sample_data,]
testset <- data[-sample_data,]
trainingset$y.class[trainingset$y==1]="yes"
trainingset$y.class[trainingset$y==0]="no"
trainingset$y<-NULL
trainingset$y.class=as.factor(trainingset$y.class)
my_control <- trainControl(method='cv',number=100,,savePredictions=TRUE,
classProbs=TRUE,sampling ="smote",summaryFunction=twoClassSummary)
library('rpart')
library('caretEnsemble')
model_list <- caretList(
  Cancellation.class~., data=trainingset,
  trControl=fitControl,
  methodList=c('LogitBoost', 'knn','AdaBoost.M1','rf','gbm'),metric="Spec",tuneLength=3)
gbm_ensemble <- caretStack(
  model_list, 
  method='gbm',
  verbose=FALSE,
  tuneLength=10,
  metric='ROC',
  trControl=trainControl(
    method='boot',
    number=4,
    savePredictions=TRUE,
    classProbs=TRUE,
    sampling = 'smote',
    summaryFunction=twoClassSummary
  )
)
ens_preds <- predict(gbm_ensemble, newdata=testset)
testset1$ensemble <- ens_preds
testset1$Cancellation=testset$Cancellation
optimal.cutpoint <- optimal.cutpoints(X = ensemble~Cancellation, tag.healthy 
= 0,  methods = "MaxEfficiency", data = testset1)
summary(optimal.cutpoint)
testset12 <- lapply(model_list, predict, newdata=trainingset, type='prob')
testset12 <- lapply(testset12, function(x) x[,2])
testset12 <- data.frame(testset12)
ens_preds <- predict(gbm_ensemble, newdata=trainingset)
testset12$ensemble <- ens_preds
testset1$y[testset$y=="0"]="no"
testset1$y[testset$y=="1"]="yes"


model_preds3 <- testset1
model_preds3$ensemble <- predict(gbm_ensemble, newdata=testset, type='raw')