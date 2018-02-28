# ABHINAV SACHDEVA
# DATA CHALLENGE

###########################
#Load necessary Libraries:
###########################

library('dplyr')
library('readr')
library('ggplot2')
library('caret')
library('glmnet')
library('rpart')
library('randomForest')
library('gbm')
library('caret')
library('xgboost')


#######################
# Load Data File: 
#######################

# I converted the data files passed on to .Rdata format

load("~/Desktop/Abhinav_Sachdeva_Data_Challenge_TMS.Rdata")

# Let's combine the train features given and the respective salaries with respect to the jobID
train_complete <- inner_join(x = train_features, y = train_salaries, by = "jobId")

#######################
# Data Exploration: 
#######################

str(train_complete)

#Let's see if all the jobIDs are actually unique or not

length(unique(train_complete$jobId))

# There you go, that means 234 jobIDs were actually duplicates

#######################
# Cleaning the Data Set:
######################

# companyID is read as a character, but ideally should be a factor. Let's convert it into a factor variable

train_complete$companyId <- as.factor(train_complete$companyId)

# Let's explore the categorical variables and see the count of the different levels

# 1. CompanyID

table(train_complete$companyId)

# 2. Job Type

table(train_complete$jobType)

# 3. Degree

table(train_complete$degree)

# 4. Major

table(train_complete$major)

# 5. Industry

table(train_complete$industry)

#####################
# Bivariate Analysis:
#####################

# Overall, it looks like all of the categorical variables have almost distributions for each of their levels

# Let's explore the categorical variables graphically as well with respect to our dependent variable: Salary

# Exploring Salaries by JobType
  ggplot(train_complete, aes(x=reorder(jobType,salary, FUN = median), salary)) + geom_boxplot(aes(fill = jobType)) + 
  labs(title="Distribution of Salaries by Job Type", x="Job Type", y="Salary") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.ticks = element_blank(), axis.text.x = element_blank()) + xlab('Job Type Ordered by Salary') +
  scale_fill_discrete(breaks = levels(reorder(train_complete$jobType, train_complete$salary, FUN = median)))

# As expected, the CEO on an average is having the highest salary

# Exploring Salaries by Degree Earned
    ggplot(train_complete, aes(x=reorder(degree,salary, FUN = median), salary)) + geom_boxplot(aes(fill = degree)) + 
    labs(title="Distribution of Salaries by Degree Earned", x="Degree Earned", y="Salary") +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.ticks = element_blank(), axis.text.x = element_blank()) + xlab('Degree Ordered by Salary') +
    scale_fill_discrete(breaks = levels(reorder(train_complete$degree, train_complete$salary, FUN = median)))
    
# Again, as expected, individuals with no degrees on an average are earning the least salary whereas the ones with a doctoral degrees have the highest salaries

# Exploring Salaries by Industry
    
      ggplot(train_complete, aes(x=reorder(industry,salary, FUN = median), salary)) + geom_boxplot(aes(fill = industry)) +
      labs(title="Distribution of Salaries by Industry", x="Industry", y="Salary") +
      theme(plot.title = element_text(hjust = 0.5)) +
      theme(axis.ticks = element_blank(), axis.text.x = element_blank()) + xlab('Industries Ordered  by Salary') +
      scale_fill_discrete(breaks = levels(reorder(train_complete$industry, train_complete$salary, FUN = median)))    
      
# Interestingly, individuals in education industry are earning the least salaries. Might be the case of probably Teaching Assistants/Graduate Assistants/Research Assistants who might be pursuing some degree education
      
    
# Exploring Salaries by Major 
      
      ggplot(train_complete, aes(x=reorder(major,salary, FUN = median), salary)) + geom_boxplot(aes(fill = major)) + 
        labs(title="Distribution of Salaries by Industry", x="Major", y="Salary") +
        theme(plot.title = element_text(hjust = 0.5)) +
        theme(axis.ticks = element_blank(), axis.text.x = element_blank()) + xlab('Major Ordered by Salary') +
        scale_fill_discrete(breaks = levels(reorder(train_complete$major, train_complete$salary, FUN = median)))      

# No real surprises here. Engineering majors are earning the highest salaries on an average!
      
# Let's check if there's a difference in the average salaries by Company
      
        train_complete %>% group_by(companyId) %>% summarise(average=mean(salary)) %>%  
        ggplot(aes(x=reorder(companyId,average), y=average)) + geom_point() +
        labs(title="Distribution of Salaries by Companies", x="Company", y="Salary") +
        theme(plot.title = element_text(hjust = 0.5)) +
        theme(axis.ticks = element_blank(), axis.text.x = element_text(angle=90, vjust=0.5, size=6)) +
        xlab('Companies Ordered by Salary') 
     

# This is interesting. There's absolutely no real difference in the average salaries offered by different companies in this dataset
# The lowest average salary is about 115 and the highest average salary being 116.8. So, no real difference to be honest
        
        
#######################
# Univariate Analysis: 
######################
        
# EXPLORE OUR DEPENDENT VARIABLE : Salary

hist(train_complete$salary, main="Distribution of Salary", 
     col="steelblue", freq=F, xlab = "Salary")
lines(density(train_complete$salary, na.rm = T), col="orange", lwd=3)

# Normally distributed, which is good news because that means we can easily perform multiple regression on it later without performing any transformation to make our dependent variable-Normal

# Let's look into the other 2 numeric variables: yearsExperience & milesFromMetropolis

# 1. yearsExperience

hist(train_complete$yearsExperience, main="Distribution of Years of Experience", 
     col="steelblue", freq=F, xlab = "Years of Experience")
lines(density(train_complete$yearsExperience, na.rm = T), col="orange", lwd=3)

# Again, we get almost a uniform distribution with only YearsOfExperience=0 being the dominant ones

# 2. milesFromMetropolis

hist(train_complete$milesFromMetropolis, main="Distribution of Miles From Metropolis", 
     col="steelblue", freq=F, xlab = "Miles From Metropolis")
lines(density(train_complete$milesFromMetropolis, na.rm = T), col="orange", lwd=3)

# Again, almost a uniform distribution. This is incredible!

#####################
# Correlation: 
#####################

# Since our Dependent variable is numeric and we have 2 numeric variables with us. Let's try to see if there's a correlation between those

# Correlation between Years of Experience & Salary

cor.test(train_complete$salary,train_complete$yearsExperience)

# Graphically
ggplot(train_complete, aes(yearsExperience, salary)) + geom_smooth()    

# Definitely, a decent positive correlation as we would expect

# Correlation between Miles From Metropolis & Salary

cor.test(train_complete$salary,train_complete$milesFromMetropolis)

# Graphically
ggplot(train_complete, aes(milesFromMetropolis, salary)) + geom_smooth() 

# Again, a decent negative correlation as expected. We are going well here...


str(training)

training$yearsExperience <- as.numeric(training$yearsExperience)
training$milesFromMetropolis <- as.numeric(training$milesFromMetropolis)
training$salary <- as.numeric(training$salary)


# We earlier saw that there's no much difference in the average salaries with respect to the different companies
# Does it have low variance? Let's check out

#################################
# Feature Engineering:
################################

train_complete %>% mutate(scaling=scale(salary, center=T, scale=T)) %>% group_by(companyId) %>% mutate(diff=mean(scaling)) %>%
  select(companyId, diff)

# Let me pick any 4 companies and check the variance. I'm picking COMP37, COMP52, COMP24, COMP7

train_complete %>% filter(companyId %in% c('COMP37','COMP52','COMP24','COMP7')) %>% 
  ggplot(aes(x=milesFromMetropolis,fill=companyId)) + geom_density(alpha=0.3)

train_complete %>% filter(companyId %in% c('COMP37','COMP52','COMP24','COMP7')) %>% 
  ggplot(aes(x=yearsExperience,fill=companyId)) + geom_density(alpha=0.3)

train_complete %>% filter(companyId %in% c('COMP37','COMP52','COMP24','COMP7')) %>% 
  ggplot(aes(x=salary,fill=companyId)) + geom_density(alpha=0.3)

train_complete %>% filter(companyId %in% c('COMP37','COMP52','COMP24','COMP7')) %>%  
  ggplot(aes(x=jobType,fill=companyId)) + geom_bar() +
  theme(axis.ticks = element_blank(), axis.text.x = element_text(angle=90, vjust=0.5, hjust=1, size=7))

train_complete %>% filter(companyId %in% c('COMP37','COMP52','COMP24','COMP7')) %>%  
  ggplot(aes(x=industry,fill=companyId)) + geom_bar() +
  theme(axis.ticks = element_blank(), axis.text.x = element_text(angle=90, vjust=0.5, hjust=1, size=7))

# Brilliant. The plots clearly show that there's low variance in the companyID variable.
# If that's the case, then there's not need to use this variance while building our model as it won't contribute much to it to predict salary which is what we want to do

#################################
# MODEL BUILDING:
################################
# DATA SPLIT- MODEL VALIDATION

set.seed(123)
in_training <- createDataPartition(train_complete$salary, p = 0.8, list = FALSE)
training <- train_complete[in_training,]
testing <- train_complete[-in_training,]

# Since our Dependent Variable is a Continous one, let's start with Multiple Regression

# MODEL 1: MULTIPLE REGRESSION

lm.fit <- lm(salary ~ jobType + degree + major + industry + yearsExperience + milesFromMetropolis, training)
lm.fitted <- predict(lm.fit, testing[, -c(1,2,9)]) # No need of JobID, CompanyID and ofcourse salary
summary(lm.fit)

# RMSE

sqrt(mean((lm.fitted - testing$salary)^2))

# RMSE = 19.59779

# Let's try Ridge & Lasso to see if we can decrease the RMSE
# Given the looks of the data, I doubt there would be much improvement, but let's check it out

# RIDGE REGRESSION 

set.seed(123)
x1 <- model.matrix(salary ~ jobType + degree + major + industry + yearsExperience + milesFromMetropolis-1, training)
y1 <- training$salary
x2 <- model.matrix(salary ~ jobType + degree + major + industry + yearsExperience + milesFromMetropolis-1, testing)
y2 <- testing$salary
cv.ridge <- cv.glmnet(x1, y1, alpha=0, nfolds=10, lambda = 10^seq(0, -5, length=100))
plot(cv.ridge)

# Let's get the optimum lambda value
bestlambda.ridge <- cv.ridge$lambda.min
bestlambda.ridge

ridge.model <- glmnet(x1, y1, alpha =0, lambda = 10^seq(0, -5, length=100), thresh =1e-12)
ridge.pred <- predict(ridge.model, s = bestlambda.ridge, newx = x2)
sqrt(mean((ridge.pred - y2)^2))

# RMSE = 19.59779. No Improvement at all !

# LASSO REGRESSION

set.seed(123)
x1 <- model.matrix(salary ~ jobType + degree + major + industry + yearsExperience + milesFromMetropolis-1, training)
y1 <- training$salary
x2 <- model.matrix(salary ~ jobType + degree + major + industry + yearsExperience + milesFromMetropolis-1, testing)
y2 <- testing$salary
lasso.model =glmnet(x1, y1, alpha = 1, lambda = 10^seq(1, -4, length = 100))
plot(lasso.model)

cv.lasso <- cv.glmnet(x1, y1, alpha= 1, nfolds = 10, lambda = 10^seq(1, -4, length = 100), thresh = 1e-12)
plot(cv.lasso)

# Let's look for the optimum value of lambda
bestlambda.lasso <- cv.lasso$lambda.min
bestlambda.lasso

lasso.pred <- predict(lasso.model, s = bestlambda.lasso, newx = x2)
sqrt(mean((lasso.pred - y2)^2))

# RMSE = 19.59779. As expected !

# MODEL 2: Random Forest

set.seed(123)
one <- training %>% sample_frac(0.2)
forest <- randomForest(salary ~ jobType + degree + major + industry + yearsExperience + milesFromMetropolis, 
                       data = one, ntree = 50, mtry = 6, nodesize = 50, importance = TRUE)

yhat <- predict(forest, newdata = testing[, -c(1,2,9)])
sqrt(mean((yhat - testing$salary)^2))

# RMSE = 19.32907. Surely an improvement from Multiple Regression. Could be better if I increase the number of trees but the computational time will be more

# Let's look into which variable is the best predictor for predicting salary

importance(forest)

# JobType by a long way followed by years of Experience

varImpPlot(forest, main = "Variable Importance")

# MODEL 3: GBM

set.seed(123)

boost <- gbm(salary ~ jobType + degree + major + industry + yearsExperience + milesFromMetropolis, 
             data=training, distribution="gaussian", n.trees = 200, interaction.depth = 3, shrinkage = 1,
             verbose = F, cv.folds = 5, bag.fraction = 0.8)

summary(boost)

y.boost <- predict(boost, newdata = testing[,-c(1,2,9)])

sqrt(mean((y.boost - testing$salary)^2))

# RMSE = 18.90163. Some improvement

# MODEL 4: XGBoost 

# Let's tune our model parameters to get a lesser RMSE 

set.seed(123)
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 2, number = 5)
xgb.grid <- expand.grid(eta = 1, max_depth = c(3,6), gamma = 0, nrounds = 2, colsample_bytree = 1, subsample = 1,
                        min_child_weight = c(10, 50))

xgb_tune <-train(salary ~ jobType + degree + major + industry + yearsExperience + milesFromMetropolis,
                 data=one,
                 method="xgbTree",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid,
                 verbose=F,
                 metric='RMSE')

trellis.par.set(caretTheme())
ggplot(xgb_tune, metric = "RMSE")

# Need to convert yearsExperience & milesFromMetropolis to numeric, otherwise xgboost won't work 

training$yearsExperience <- as.numeric(training$yearsExperience)
training$milesFromMetropolis <- as.numeric(training$milesFromMetropolis)

testing$yearsExperience <- as.numeric(testing$yearsExperience)
testing$milesFromMetropolis <- as.numeric(testing$milesFromMetropolis)


dx.train <- training[,c(3:8)] %>% data.matrix 
dx.test  <- testing[,c(3:8)] %>% data.matrix

bst <- xgboost(data = dx.train, label = training$salary, max.depth = 3, eta = 1, nround = 500, 
               objective = "reg:linear", folds=5, sub_sample=0.8, min_child_weight = 10)

y.bst <- predict(bst, newdata = dx.test)
sqrt(mean((y.bst - testing$salary)^2))

# RMSE = 18.87. Even better

importance_matrix <- xgb.importance(attributes(dx.train)$dimnames[[2]], model = bst)
xgb.plot.importance(importance_matrix)


# MODEL 5: Ensemble(GBM & XGBoost)

avg <- (y.boost + y.bst)/2
sqrt(mean((avg - testing$salary)^2))

# Test Dataset
str(test_features)

# Let's correct the datatype of variables in our test dataset

test_features$companyId <- as.factor(test_features$companyId)
test_features$jobType <- as.factor(test_features$jobType)
test_features$degree <- as.factor(test_features$degree)
test_features$major <- as.factor(test_features$major)
test_features$industry <- as.factor(test_features$industry)
test_features$yearsExperience <- as.numeric(test_features$yearsExperience)
test_features$milesFromMetropolis <- as.numeric(test_features$milesFromMetropolis)

# Looks good now

# Let's try to train our model with GBM first
set.seed(123)
boost.final <- gbm(salary ~ jobType + degree + major + industry + yearsExperience + milesFromMetropolis, 
                   data=train_complete, distribution="gaussian", n.trees = 200, interaction.depth = 3, shrinkage = 1,
                   verbose = F, cv.folds = 5, bag.fraction = 0.8)

x.train <- train_complete[,c(3:8)] %>% data.matrix 

# Now, let's do it with XGBoost

str(train_complete)

train_complete$yearsExperience <- as.numeric(train_complete$yearsExperience)
train_complete$milesFromMetropolis <- as.numeric(train_complete$milesFromMetropolis)
train_complete$salary <- as.numeric(train_complete$salary)


bst.final <- xgboost(data = x.train, label = train_complete$salary, max.depth = 3, eta = 1, nround = 500, 
                     objective = "reg:linear", folds=5, sub_sample=0.8, min_child_weight = 10)

y.boost.test <- predict(boost.final, newdata = test_features[,3:8])
y.bst.test <- predict(bst.final, newdata = data.matrix(test_features[,3:8]))

# Predictions by emsemble method (GBM & XBBoost)
prediction <- data.frame(jobId = test_features$jobId, salary = (y.boost.test + y.bst.test)/2)

# Exporting our predictions to the test_salaries csv

prediction %>% write_csv('test_salaries.csv')


save.image(file = "Abhinav_Sachdeva_Data_Challenge_TMS_Final.Rdata")
