library('caret')
library(DMwR)
dataset<-read.csv(file.choose(),header = TRUE)
dataset$Freq_Dummy<-as.factor(dataset$Freq_Dummy)
 dataset$Cab_Model<-as.factor(dataset$Cab_Model)
 dataset$Area_Dummy<-as.factor(dataset$Area_Dummy)
 dataset$Travel.Type<-as.factor(dataset$Travel.Type)
 dataset$Month_Date<-as.factor(dataset$Month_Date)
 dataset$Weekday_date<-as.factor(dataset$Weekday_date)
 dataset$month.date<-as.factor(dataset$month.date)
 dataset$Weekday_Booking<-as.factor(dataset$Weekday_Booking)
 dataset$Time_Dummy<-as.factor(dataset$Time_Dummy)
 dataset$Month_Booking<-as.factor(dataset$Month_Booking)
 dataset$Online_Booking<-as.factor(dataset$Online_Booking)
 dataset$Mobile_Site_Booking<-as.factor(dataset$Mobile_Site_Booking)
dataset$Weekday_booking<-as.factor(dataset$Weekday_booking)
dataset$Cancellation<-as.factor(dataset$Cancellation)
set.seed(1234)
sample_data <- createDataPartition(y = dataset$Cancellation, p = .70, list = FALSE)
trainingset <- dataset[ sample_data,]
testset <- dataset[-sample_data,]
predictdummy.tree= prediction(prob.tree,trainingset$admit)


fitControl <- trainControl(method = "cv",number = 10,repeats = 5,classProbs = TRUE,
                           summaryFunction = twoClassSummary)

trainingset$Cancellation.class[trainingset$Cancellation==1]="yes"
trainingset$Cancellation.class[trainingset$Cancellation==0]="no"
trainingset$Cancellation<-NULL
fit.rpart<-train(Cancellation.class~.,trainingset,method='rf',trControl =fitControl,metric="ROC",tunelength=5)
trainingset$Cancellation.class<-as.factor(trainingset$Cancellation.class)


new<-randomForest(Cancellation~.,data=Dummy)
prob.tree<-predict(new,testset,type="prob")
prob.tree<-prob.tree[1:9120]
prob.tree<-1-prob.tree
testset1<-data.frame(S.no=1:9120)
testset1$Cancellation<-testset$Cancellation
optimal.cutpoint <- optimal.cutpoints(X = prob~Cancellation, tag.healthy = 0,
control=control.cutpoints(costs.ratio = 100),methods = "CB", data = testset1)
summary(optimal.cutpoint)
testset1$predict=0
testset1$predict[testset1$prob>.3]=1
confusionMatrix(testset1$predict,testset1$Cancellation)




prf2 =performance(predictdummy.tree,"tpr","fpr")
plot( prf2, colorize = FALSE)
plot(prf, add = TRUE, colorize = FALSE)