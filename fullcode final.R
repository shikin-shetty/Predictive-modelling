library(glmnet)
library(leaps)
library(randomForest)
library(gbm)
library(pls)
library(MASS)
library(class)
library(adabag)
library(caret)
#Regression

##ONFIELD PLAYERS
#Load the CSV file
data <- read.csv(file.choose())

#Remove the unnecessary columns
fifa <- data[-c(1,3:6,8,9,12:19)]

##Split into Goal keeper and Field Player
fifa.gk <- fifa[which(fifa$prefers_gk=="True"),]
fifa.fp <- fifa[which(fifa$prefers_gk=="False"),]

##ONFIELD PLAYER
##Remove GK column
fifa.fp <- fifa.fp[-79]
##Remove Columns with only one level
fifa.fp <-Filter(function(x)(length(unique(x))>1), fifa.fp)
fifa.fp <-  na.omit(fifa.fp)
fifa.fp <-Filter(function(x)(length(unique(x))>1), fifa.fp)
##Remove name column
fifa.fp1 <- fifa.fp[-1]

##ONLY NUMERICAL PREDICTORS CONSIDERED FOR REGRESSION
fifa.fp2 <- fifa.fp1[-c(15:17,78:133)]

numpred=ncol(fifa.fp2)-1
fifa.fptemp=fifa.fp2[-c(4)]
fifa.fptemp
zscaletrain=preProcess(fifa.fptemp[,1:numpred])
scaledx=predict(zscaletrain,fifa.fptemp[,1:numpred])
corrpred=findCorrelation(cor(scaledx),cutoff=0.8)
uncor=scaledx[,-corrpred]
pcatrain=preProcess((uncor),method = "pca",thresh=0.8)
pcatrain

set.seed(3)
pr.out=prcomp(fifa.fptemp[1:numpred],scale=TRUE)
pr.out$rotation
names(pr.out)
pr.out$centre
pr.out$scale
pr.out$rotation
biplot(pr.out,scale=0)
pr.out$sdev
pr.var=pr.out$sdev^2
pr.var  
pve=pr.var/sum(pr.var)
sum=0
pve

#cumulative sum of variance ratios
for (i in 1:numpred )
{
  if (i==1)
  {sum[1]=pve[i]}
  else
  {sum[i]=sum[i-1]+pve[i]}
}
sum[1:numpred]

#calculating pve indices for different % variances
varlim=0.8
j=0
for ( i in 2:numpred-1)
{
  if (sum[i]>=varlim && sum[i-1]<varlim)
  {j=i}
  else
  {j=j}
}
j

##SPLIT INTO TRAINING AND TEST DATA - ONLY NUMERICAL PREDICTORS
set.seed(3)
train.fp <- sample(1:nrow(fifa.fp2),nrow(fifa.fp2)-100)
test.fp <- fifa.fp2[-train.fp,]

##LINEAR MODEL
lm.fp <- lm(overall~.,data=fifa.fp2,subset=train.fp)
lm.predict <- predict(lm.fp,new=test.fp)
msefp <- mean((lm.predict-test.fp$overall)^2)
#The Mean Square Error is
msefp

##BACKWARD ELIMINATION WITH TRAINING DATA

regfit.bwd <- regsubsets(overall~.,data=fifa.fp2[train.fp,],nvmax=73,method="backward")
reg.summary <- summary(regfit.bwd)
reg.summary
#model matrix for test
test.mat <- model.matrix(overall~.,data=test.fp)
#Compute test mse
val.errors <- rep(NA,57)
for (i in 1:57){
  coefi <- coef(regfit.bwd,id=i)
  pred <- test.mat[,names(coefi)]%*%coefi
  val.errors[i] <- mean((test.fp$overall-pred)^2)
}
val.errors

#Find the best model
which.min(val.errors)
coef(regfit.bwd,30)

#Using adjR2
reg.summary$adjr2
which.max(reg.summary$adjr2)
#Using Cp
reg.summary$cp
which.min(reg.summary$cp)
#Using BIC
reg.summary$bic
which.min(reg.summary$bic)


##LASSO
#Fit the lasso with training data

x <- model.matrix(overall~.,fifa.fp2)[,-1]
y <- fifa.fp2$overall
lasso.mod <- glmnet(x[train.fp,],y[train.fp],alpha=1)
#Use CV to find optimal lambda
set.seed(1)
cv.out <- cv.glmnet(x[train.fp,],y[train.fp],alpha=1)
plot(cv.out)
bestlam <- cv.out$lambda.min
#Refit the lasso with the best lamda
lasso.mod <- glmnet(x[train.fp,],y[train.fp],alpha=1,lambda=bestlam)
#Predict The Coefficient estimates
lasso.pred <- predict(lasso.mod,s=bestlam,newx=x[-train.fp,])
#MSE OF LASSO
mean((lasso.pred-y[-train.fp])^2)

##RANFDOM FORESTS
p <- round((ncol(fifa.fp2)-1)/3)
p
set.seed(3)
rf.fifafp <- randomForest(overall~.,data=fifa.fp2,subset=train.fp,mtry=p,ntree=100,
                          importance=TRUE)
#Predict the test data
test.rf <- predict(rf.fifafp,newdata=test.fp)
#TEST MSE
mean((test.rf - test.fp[,"overall"])^2)
#Which Variables are important
varImpPlot(rf.fifafp)

#PCR

require(pls)
pcr_modelfp <- pcr(overall~.,data=fifa.fp2,subset=train.fp,scale=TRUE, validation="CV")
pcr_predfp <- predict(pcr_modelfp,newdata=test.fp,ncomp=21)
summary(pcr_modelfp)
mean((pcr_predfp - test.fp[,"overall"])^2)

#BOOSTING
set.seed(3)
boost.fifafp <- gbm(overall~.,data=fifa.fp2[train.fp,],distribution="gaussian",
                    n.trees=100,interaction.depth=1)
summary(boost.fifafp)
par(nfrow=c(1,2))
plot(boost.fifafp,i="reactions")
plot(boost.fifafp,i="potential")
boost.fifafp.pred <- predict(boost.fifafp,newdata=test.fp,n.trees=100)
#MSE
mean((boost.fifafp.pred-test.fp$overall)^2)










#GOALKEEPER CODE



##GOALKEEPER PLAYER
head(fifa.gk)
colnames(fifa.gk)
fifa.gk <- fifa.gk[-79]
fifa.gk <-Filter(function(x)(length(unique(x))>1), fifa.gk)
head(fifa.gk)
fifa.gk <-  na.omit(fifa.gk)
head(fifa.gk)
fifa.gk <-Filter(function(x)(length(unique(x))>1), fifa.gk)
fifa.gk1 <- fifa.gk[,-1]
head(fifa.gk1)

ncol(fifa.gk1)

colnames(fifa.gk1)
head(fifa.gk1)
#Removing classification vectors
fifa.gk2 <- fifa.gk1[,-c(14,49:64)]
head(fifa.gk2)

numpred=ncol(fifa.gk2)-1
fifa.gktemp=fifa.gk2[-c(4)]
fifa.gktemp
zscaletrain=preProcess(fifa.gktemp[,1:numpred])
scaledx=predict(zscaletrain,fifa.gktemp[,1:numpred])
corrpred=findCorrelation(cor(scaledx),cutoff=0.8)
uncor=scaledx[,-corrpred]
pcatrain=preProcess((uncor),method = "pca",thresh=0.8)
pcatrain

set.seed(3)
pr.out=prcomp(fifa.gktemp[1:numpred],scale=TRUE)
pr.out$rotation
names(pr.out)
pr.out$centre
pr.out$scale
pr.out$rotation
biplot(pr.out,scale=0)
pr.out$sdev
pr.var=pr.out$sdev^2
pr.var  
pve=pr.var/sum(pr.var)
sum=0
pve

#cumulative sum of variance ratios
for (i in 1:numpred )
{
  if (i==1)
  {sum[1]=pve[i]}
  else
  {sum[i]=sum[i-1]+pve[i]}
}
sum[1:numpred]

#calculating pve indices for different % variances
varlim=0.8
j=0
for ( i in 2:numpred-1)
{
  if (sum[i]>=varlim && sum[i-1]<varlim)
  {j=i}
  else
  {j=j}
}
j


##SPLIT INTO TRAINING AND TEST DATA - ONLY NUMERICAL PREDICTORS
set.seed(3)
train.gk <- sample(1:nrow(fifa.gk2),nrow(fifa.gk2)-100)
test.gk <- fifa.gk2[-train.gk,]

##LINEAR MODEL
lm.gk <- lm(overall~.,data=fifa.gk2,subset=train.gk)
summary(lm.gk)
lm.predictgk <- predict(lm.gk,new=test.gk)
msegk <- mean((lm.predictgk-test.gk$overall)^2)
#The Mean Square Error is
msegk
#3.48

##BACKWARD ELIMINATION WITH TRAINING DATA
ncol(fifa.gk2)
set.seed(3)
regfit.bwdgk <- regsubsets(overall~.,data=fifa.gk2[train.gk,],nvmax=54,method="backward")
reg.summarygk <- summary(regfit.bwdgk)
reg.summarygk
#model matrix for test
test.matgk <- model.matrix(overall~.,data=test.gk)
#Compute test mse
val.errorsgk <- rep(NA,1)
for (i in 1:41){
  coefigk <- coef(regfit.bwdgk,id=i)
  predgk <- test.matgk[,names(coefigk)]%*%coefigk
  val.errorsgk[i] <- mean((test.gk$overall-predgk)^2)
}
val.errorsgk
#3.269

#Find the best model
which.min(val.errorsgk)
coef(regfit.bwdgk,which.min(val.errorsgk))

#Using adjR2
reg.summarygk$adjr2
which.max(reg.summarygk$adjr2)
#35
#Using Cp
reg.summarygk$cp
which.min(reg.summarygk$cp)
#28
#Using BIC
reg.summarygk$bic
which.min(reg.summarygk$bic)
#14

#LASSO
#Fit the lasso with training data

x <- model.matrix(overall~.,fifa.gk2)[,-1]
x
y <- fifa.gk2$overall
lasso.mod <- glmnet(x[train.gk,],y[train.gk],alpha=1)
#Use CV to find optimal lambda
set.seed(3)
cv.outgk <- cv.glmnet(x[train.gk,],y[train.gk],alpha=1)
plot(cv.outgk)
bestlamgk <- cv.outgk$lambda.min
#Refit the lasso with the best lamda
lasso.modgk <- glmnet(x[train.gk,],y[train.gk],alpha=1,lambda=bestlamgk)
#Predict The Coefficient estimates
lasso.predgk <- predict(lasso.modgk,s=bestlamgk,newx=x[-train.gk,])
#MSE OF LASSO
mean((lasso.predgk-y[-train.gk])^2)
#2.826
coef(lasso.modgk)

#RANFDOM FORESTS

p <- round((ncol(fifa.gk2)-1)/3)
p

set.seed(3)
rf.fifagk <- randomForest(overall~.,data=fifa.gk2,subset=train.gk,mtry=p,ntree=100,
                          importance=TRUE)

#Predict the test data

test.rfgk <- predict(rf.fifagk,newdata=test.gk)
#TEST MSE
mean((test.rfgk - test.gk[,"overall"])^2)
#1.884
#Which Variables are important
rf.fifagk$importance

varImpPlot(rf.fifagk)

#PCR

require(pls)
pcr_modelgk <- pcr(overall~.,data=fifa.gk2,subset=train.gk,scale=TRUE, validation="CV")
summary(pcr_modelgk)
pcr_predgk <- predict(pcr_modelgk,newdata=test.gk,ncomp=21)
mean((pcr_predgk - test.gk[,"overall"])^2)
#0.244

#BOOSTING


set.seed(3)
boost.fifagk <- gbm(overall~.,data=fifa.gk2[train.gk,],distribution="gaussian",
                    n.trees=100,interaction.depth=1)
summary(boost.fifagk)
boost.fifagk.pred <- predict(boost.fifagk,newdata=test.gk,n.trees=100)
#MSE
mean((boost.fifagk.pred-test.gk$overall)^2)
#2.59

##CLASSIFICATION

##FP ATTRIBUTES CLASSIFICATION


##DATA SET WITH AGGREGATED POSITIONS
fifa.agg <- fifa.fp1[,c(120:133)]
fifa.agg$position[fifa.agg$prefers_cf == "True"] <- "Striker"
fifa.agg$position[fifa.agg$prefers_cam == "True"] <- "Striker"
fifa.agg$position[fifa.agg$prefers_cb == "True"] <- "Defender"
fifa.agg$position[fifa.agg$prefers_cdm == "True"] <- "Midfielder"
fifa.agg$position[fifa.agg$prefers_cm == "True"] <- "Midfielder"
fifa.agg$position[fifa.agg$prefers_lb == "True"] <- "Defender"
fifa.agg$position[fifa.agg$prefers_lm == "True"] <- "Midfielder"
fifa.agg$position[fifa.agg$prefers_lw == "True"] <- "Striker"
fifa.agg$position[fifa.agg$prefers_lwb == "True"] <- "Defender"
fifa.agg$position[fifa.agg$prefers_rm == "True"] <- "Midfielder"
fifa.agg$position[fifa.agg$prefers_rw == "True"] <- "Striker"
fifa.agg$position[fifa.agg$prefers_rwb == "True"] <- "Defender"
fifa.agg$position[fifa.agg$prefers_st == "True"] <- "Striker"
fifa.agg$position[fifa.agg$prefers_rb == "True"] <- "Defender"

##CLASSIFICATION DATA SET
fifa.cl <- fifa.fp1[,-c(120:133)]
fifa.cl$position <- fifa.agg$position
fifa.cl$position <- as.factor(fifa.cl$position)

##SPLIT INTO TRAINING AND TEST DATA SET
set.seed(3)
train <- sample(1:nrow(fifa.cl),nrow(fifa.cl)-100)
test <- fifa.cl[-train,]
fifa.cl <- na.omit(fifa.cl)

##RANDOM FORESTS
#No of predictors
p <- ncol(fifa.cl)-1
p <- round(sqrt(p))

set.seed(3)
rf.fifacl <- randomForest(position ~ ., data= fifa.cl,
                          subset = train, mtry=p, ntree=100,
                          importance=TRUE)
pred.rf <- predict(rf.fifacl,newdata=fifa.cl[-train,],type="class")
tab <- table(pred.rf,test$position)
##TABLE
tab
##TEST ERROR RATE
test.error.rate.rf <- mean(pred.rf!=test$position)
test.error.rate.rf
##VARIABLES IMPORTANCE
varImpPlot(rf.fifacl)

##LDA

lda.fit <- lda(position~.,data=fifa.cl,subset=train)
lda.pred <- predict(lda.fit,test)
lda.class <- lda.pred$class
#TABLE
table(lda.class,test$position)
#TEST ERROR RATE
test.error.rate.lda <- mean(lda.class!=test$position)
test.error.rate.lda



##KNN
##CONVERT IN NUMERICAL VALUES
fifa.num <- fifa.cl
for(i in 78:119)
{
  fifa.num[,i] <- as.integer(fifa.num[,i]=="True")
}
lookup <- c("Right"=1,"Left"=0)
fifa.num$preferred_foot <- lookup[fifa.cl$preferred_foot]
lookup1 <- c("High"=3,"Medium"=2,"Low"=1)
fifa.num$work_rate_att <- lookup1[fifa.cl$work_rate_att]
fifa.num$work_rate_def <- lookup1[fifa.cl$work_rate_def]
fifa.num$position <- fifa.cl$position
##SPLIT THIS DATA
set.seed(3)
train <- sample(1:nrow(fifa.num),nrow(fifa.num)-100)
test <- fifa.num[-train,]
test.error.rate.knn <- c()

for(i in 1:100)
{
  set.seed(3)
  knn.cl <- knn(fifa.num[train,-120],fifa.num[-train,-120],fifa.num[train,]$position,k=i)
  tab <- table(knn.cl,test$position)
  #ERROR RATE
  test.error.rate.knn[i] <-  mean(knn.cl!=test$position)
}
test.error.rate.knn
which.min(test.error.rate.knn)
##LEAST ERROR RATE - 0.14 FOR K = 20
##REVISIT KNN FOR K = 20
knn.cl <- knn(fifa.num[train,-120],fifa.num[-train,-120],fifa.num[train,]$position,k=20)
table(knn.cl,test$position)
mean(knn.cl!=test$position)

##BOOSTING
##USING ADABOOST

set.seed(3)
fifa.adaboost <- boosting(position ~.,data=fifa.num[train, ], mfinal=100, coeflearn="Zhu",
                          control=rpart.control(maxdepth=1))
fifa.adaboost.pred <- predict.boosting(fifa.adaboost,
                                       newdata=fifa.num[-train, ])
##CONFUSITON MATRIX TABLE
fifa.adaboost.pred$confusion
##TEST ERROR RATE
fifa.adaboost.pred$error
##Compare error evloution in training and test set
errorevol(fifa.adaboost,newdata=fifa.num[train, ])->evol.train
errorevol(fifa.adaboost,newdata=fifa.num[-train, ])->evol.test
plot.errorevol(evol.test,evol.train)
