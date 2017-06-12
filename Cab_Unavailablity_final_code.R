library('caret')
library('caretList')
library(DMwR)
dataset<-read.csv(file.choose(),header = TRUE)

str(dataset)
dataset$X<-NULL
dataset$Cancellation.class[dataset$Cancellation==1]="yes"
dataset$Cancellation.class[dataset$Cancellation==0]="no"
dataset$Cancellation<-NULL
dataset$Freq_Dummy<-as.factor(dataset$Freq_Dummy)
dataset$Cab_Model<-as.factor(dataset$Cab_Model)
dataset$Area_Dummy<-as.factor(dataset$Area_Dummy)
dataset$Travel.Type<-as.factor(dataset$Travel.Type)
dataset$Weekday_date<-as.factor(dataset$Weekday_date)
dataset$month.date<-as.factor(dataset$month.date)
dataset$Weekday_Booking<-as.factor(dataset$Weekday_Booking)
dataset$Time_Dummy<-as.factor(dataset$Time_Dummy)
dataset$Month_Booking<-as.factor(dataset$Month_Booking)
dataset$Online_Booking<-as.factor(dataset$Online_Booking)
dataset$Mobile_Site_Booking<-as.factor(dataset$Mobile_Site_Booking)
dataset$Weekday_Booking<-as.factor(dataset$Weekday_Booking)
dataset$Cancellation.class<-as.factor(dataset$Cancellation.class)
set.seed(1234)
sample_data <- createDataPartition(y = dataset$Cancellation.class, p = .70, list = FALSE)
trainingset <- dataset[ sample_data,]
testset <- dataset[-sample_data,]
my_control <- trainControl(
  method='repeatedcv',
  number=2,repeats =2,
  classProbs=TRUE,
  savePredictions=TRUE,
  sampling ="smote",
  index=createResample(trainingset$Cancellation.class,2),
  summaryFunction=twoClassSummary
)

model_list_big <- caretList(
  Cancellation.class~., data=trainingset,
  trControl=my_control,
  metric='ROC',
  methodList=c('knn', 'AdaBoost.M1','rf','gbm','LogitBoost'),
  tuneList=list(
    rf1=caretModelSpec(method='rf', tuneGrid=data.frame( .mtry = 4),
                       LogitBoost=caretModelSpec(method='LogitBoost', tuneGrid=data.frame(nIter=31)),
                       Knn=caretModelSpec(method='knn', tuneGrid=data.frame(k=c(11,14))),
                       gbm=caretModelSpec(method='gbm', tuneGrid=data.frame(interaction.depth = 9,
                                                                            n.trees = 1950,
                                                                            shrinkage = .1,
                                                                            n.minobsinnode = 20),verbose=FALSE),
                       Ada=caretModelSpec(method='AdaBoost.M1', tuneGrid=data.frame(mfinal = 9,
                                                                                    maxdepth = 3,
                                                                                   coeflearn = "Freund")))))
greedy_ensemble <- caretEnsemble(model_list_big)
summary(greedy_ensemble)

### validation set

new_data <- createDataPartition(y = testset$Cancellation, p = .66, list = FALSE)
validationset <- testset[ new_data,]
testset <- testset[-new_data,]

probscore <- lapply(model_list_big, predict, newdata=validationset, type='prob')
probscore <- lapply(probscore, function(x) x[,2])
probscore <- data.frame(probscore)

ens_preds1 <- predict(greedy_ensemble, newdata=validationset)
probscore$ensemble <- ens_preds1
probscore$Cancellation=validationset$Cancellation

probscore$Cancellation.class[probscore$Cancellation=="yes"]=1
probscore$Cancellation.class[probscore$Cancellation=="no"]=0

optimal.cutpoint <- optimal.cutpoints(X = ensemble~Cancellation.class, tag.healthy 
                                      = 0,  methods = "MaxEfficiency", data = probscore)
summary(optimal.cutpoint)
probscore$predict.ensemble=0

probscore$predict.ensemble[probscore$ensemble>.453]=1
confusionMatrix(probscore$predict.ensemble,probscore$Cancellation.class)

### test set

probtest <- lapply(model_list_big, predict, newdata=testset, type='prob')
probtest <- lapply(probtest, function(x) x[,2])
probtest <- data.frame(probtest)

ens_preds1 <- predict(greedy_ensemble, newdata=testset)
probtest$ensemble <- ens_preds1
probtest$Cancellation=testset$Cancellation

probtest$Cancellation.class[probtest$Cancellation=="yes"]=1
probtest$Cancellation.class[probtest$Cancellation=="no"]=0

optimal.cutpoint <- optimal.cutpoints(X = ensemble~Cancellation.class, tag.healthy 
                                      = 0,  methods = "MaxEfficiency", data = probtest)
summary(optimal.cutpoint)
probtest$predict.ensemble=0

probtest$predict.ensemble[probtest$ensemble>.453]=1
confusionMatrix(probtest$predict.ensemble,probtest$Cancellation.class)

#Lift and Roc Chart

predictdummy.ensemble= prediction(probscore$ensemble,probscore$Cancellation)
prf.ensemble =performance(predictdummy.ensemble,"tpr","fpr")
plot(prf.ensemble, add = TRUE, col='blue')
plot("lift","rpp")

predictdummy.LogitBoost= prediction(probscore$LogitBoost,probscore$Cancellation)
prf.Logit =performance(predictdummy.LogitBoost,"lift","rpp")
plot(prf.Logit,main="lift curve",ylim=c(1, 4))

predictdummy.knn= prediction(probscore$knn,probscore$Cancellation)
prf.knn =performance(predictdummy.knn,"lift","rpp")
plot(prf.knn, add = TRUE, col='red',ylim=c(1, 4))

predictdummy.Ada= prediction(probscore$AdaBoost.M1,probscore$Cancellation)
prf.Ada =performance(predictdummy.Ada,"lift","rpp")
plot(prf.Ada, add = TRUE, col='green',ylim=c(1, 4))

predictdummy.rf= prediction(probscore$rf,probscore$Cancellation)
prf.rf =performance(predictdummy.rf,"lift","rpp")
plot(prf.rf, add = TRUE,  col='orange',ylim=c(1, 4))

predictdummy.gbm= prediction(probscore$gbm,probscore$Cancellation)
prf.gbm =performance(predictdummy.gbm,"lift","rpp")
plot(prf.gbm, add = TRUE, col='yellow',ylim=c(1, 4))

predictdummy.ensemble= prediction(probscore$ensemble,probscore$Cancellation)
prf.ensemble =performance(predictdummy.ensemble,"lift","rpp")
plot(prf.ensemble, add = TRUE, col='blue')


## score file

scoreset<-read.csv(file.choose(),header = TRUE)

str(scoreset)

scoreset$Cancellation.class[scoreset$Cancellation==1]="yes"
scoreset$Cancellation.class[scoreset$Cancellation==0]="no"


scoreset$Freq_Dummy<-as.factor(scoreset$Freq_Dummy)
scoreset$Cab_Model<-as.factor(scoreset$Cab_Model)
scoreset$Area_Dummy<-as.factor(scoreset$Area_Dummy)
scoreset$Travel.Type<-as.factor(scoreset$Travel.Type)
scoreset$Weekday_date<-as.factor(scoreset$Weekday_date)
scoreset$month.date<-as.factor(scoreset$month.date)
scoreset$Weekday_Booking<-as.factor(scoreset$Weekday_Booking)
scoreset$Time_Dummy<-as.factor(scoreset$Time_Dummy)
scoreset$Month_Booking<-as.factor(scoreset$Month_Booking)
scoreset$Online_Booking<-as.factor(scoreset$Online_Booking)
scoreset$Mobile_Site_Booking<-as.factor(scoreset$Mobile_Site_Booking)
scoreset$Weekday_Booking<-as.factor(scoreset$Weekday_Booking)
scoreset$Cancellation.class<-as.factor(scoreset$Cancellation.class)




ens_preds1 <- predict(greedy_ensemble, newdata=testset)
scoreset$ensemble <- ens_preds1
scoreset<-scoreset[,-(2:14)]
scoreset$prediction=0
scoreset$prediction[scoreset$ensemble >.453]=1
write.csv(scoreset,file="finalscore.csv")
