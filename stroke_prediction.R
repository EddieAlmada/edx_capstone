# Attribute Information
# 
# 1) id: unique identifier
# 2) gender: "Male", "Female" or "Other"
# 3) age: age of the patient
# 4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
# 5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
# 6) ever_married: "No" or "Yes"
# 7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
# 8) Residence_type: "Rural" or "Urban"
# 9) avg_glucose_level: average glucose level in blood
# 10) bmi: body mass index
# 11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
#   12) stroke: 1 if the patient had a stroke or 0 if not
# *Note: "Unknown" in smoking_status means that the information is unavailable for this patient

##########################################################
# Eduardo Almada
# 2021/03/16
# Edx: Data Science - Capstone
#   This script trains a machine learning algorithm
#   to predict a stroke
##########################################################

## Installing packages
install.packages('tidyverse') # metapackage of all tidyverse packages
install.packages('naniar') # handling missing data
install.packages('skimr')# quick overview over the dataset
install.packages('caret')# Machine learning toolkit
install.packages('MLmetrics') # F1 Score
install.packages('imbalance') # algorithms to deal with imbalanced datasets
install.packages('gridExtra') # display plots in grids
install.packages('patchwork') # arrange plots side by side
install.packages('ranger') # Method for random forest

## calling packages
library(tidyverse) 
library(naniar) 
library(skimr) 
library(caret) 
library(MLmetrics) 
library(imbalance) 
library(gridExtra) 
library(patchwork) 

# Load data from a csv within a zip file
data <- read_csv(unz('archive.zip','healthcare-dataset-stroke-data.csv'))

# data preview
head(data) # first 6 rows

summary(data)# data summary

##Formatting and cleaning the data
# looking for N/A
miss_scan_count(data = data, search = list("N/A", "Unknown"))

# replacing N/A and Unknown with NA
clean_data <- replace_with_na(data,replace = list(bmi = c('N/A'),
                            smoking_status = c("Unknown")))

#  missing values visualization
vis_miss(clean_data, cluster = TRUE)

# cleaning and formatting data
post_data <- clean_data %>% mutate(stroke = factor(stroke), 
        bmi = ifelse(is.na(bmi),median(bmi,na.rm = T),bmi),
        hypertension = factor(hypertension),
        heart_disease = factor(heart_disease)) %>% filter(!gender == 'Other') %>% 
        select(-id)

#  missing values visualization
vis_miss(post_data, cluster = TRUE)

# filling smoking_status Na's with previous values
post_data <- post_data %>% fill(smoking_status)

# clean data summary
summary(post_data)


############### plotting for visual analysis ##################
# numeric variable distribution
grid.arrange(
a = post_data %>% 
  ggplot(aes(as.numeric(bmi), fill = as.factor(stroke))) + geom_density(alpha = 0.2) +
  xlab('Body mass index') + ggtitle('Numeric Variable distribution'),
b = post_data %>% 
  ggplot(aes(avg_glucose_level, fill = stroke)) + geom_density(alpha = 0.2) +
  xlab('Average glucose level'),
a = post_data %>% 
  ggplot(aes(age, fill = stroke)) + geom_density(alpha = 0.2) +
  xlab('Age')
)

######################################################################

######## Data exploration ##########################
# counting people with stroke records
post_data %>% group_by(stroke) %>%
  summarize(n = n()) %>% mutate(prop = round(n / sum(n), 2))

# analyzing the data gender balance
table(post_data$gender)

# preparing data for ML algorithms
post_data <- post_data %>%
  mutate(across(c(hypertension, heart_disease), factor), # changing hypertension & heart_disease class to factor
         across(where(is.character), as.factor), # changing character class to factors
         across(where(is.factor), as.numeric), # changing factors to numeric
         stroke = factor(ifelse(stroke == 1, '0', '1')), # setting stroke factors to 1 for positive and 0 to negative
        # changing bmi vector from numeric to categorical according to the CDC categories
        # https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/index.html
         bmi = case_when(bmi < 18.5 ~ 'underweight',
        bmi >= 18.5 & bmi < 25 ~ 'normal_weight',
        bmi >= 25 & bmi < 30 ~ 'overweight',
        bmi >= 30 ~ 'obese'),
        bmi = factor(bmi, 
        levels = c("underweight", "normal_weight",
        "overweight", "obese"), order = TRUE)
        ) %>% as_tibble()
      
View(post_data)
###################################

########### Set training and validation set #########
# setting root for repeatability purposes
set.seed(2021) 

# creating train and test set
index <-  createDataPartition(post_data$stroke, times = 1, p = 0.3, list = F)
o_train_set <- post_data[-index,]
o_test_set <- post_data[index,]

# dimensioning sets
dim(o_train_set)
dim(o_test_set)

##################### lda model ####################
set.seed(2021) # setting root for repeatability purposes
lda_model <- train(stroke ~  ., method = 'lda', data = o_train_set)
confusionMatrix(predict(lda_model, o_test_set), o_test_set$stroke)

##################### glm model ####################
set.seed(2021) # setting root for repeatability purposes
glm_model <- train(stroke ~ ., method = 'glm', data = o_train_set)
confusionMatrix(predict(glm_model, o_test_set), o_test_set$stroke)

##################### random forest model ####################
set.seed(2021) # setting root for repeatability purposes

# setting the tune grid
rfGrid <- data.frame(.mtry = c(2,3,5,6),.splitrule = "gini",
  .min.node.size = 5)

# setting the control parameters
rfControl <- trainControl(
  method = "oob", number = 5,
  verboseIter = TRUE)

rf_model <- train(stroke ~ ., data = o_train_set,
  method = "ranger", tuneLength = 3,
  tuneGrid = rfGrid,trControl = rfControl)
confusionMatrix(predict(rf_model,o_test_set),o_test_set$stroke)

# it is clear that due to the high prevalence the specificity will
# always under perform

######### improving the prevalence ###############
# counting people with stroke records
post_data %>% group_by(stroke) %>%
  summarize(n = n()) %>% mutate(prop = round(n / sum(n), 2)) # 5% of imbalance

#Majority Weighted Minority Oversampling Technique
post_dataa <- oversample(as.data.frame(post_data), 
           classAttr = "stroke", ratio = 1, method = "MWMOTE")

# counting people with stroke records now balanced
post_dataa %>% group_by(stroke) %>%
  summarize(n = n()) %>% mutate(prop = round(n / sum(n), 2))
###################################

########### Set training and validation set pt.2 #########
# setting root for repeatability purposes
set.seed(2021) 

# creating train and test set
index <-  createDataPartition(post_dataa$stroke, times = 1, p = 0.3, list = F)
train_set <- post_dataa[-index,]
test_set <- post_dataa[index,]

# dimensioning sets
dim(train_set)
dim(test_set)

##################### glm model ####################
set.seed(2021) # setting root for repeatability purposes
glm_model <- train(stroke ~ ., method = 'glm', data = train_set)
confusionMatrix(predict(glm_model, test_set), test_set$stroke)

# Accuracy for the glm with balance data
F1_Score(test_set$stroke,predict(glm_model,test_set))

##################### random forest model ####################
set.seed(2021) # setting root for repeatability purposes

# setting the tune grid
rfGrid <- data.frame(.mtry = c(2,3,5,6),.splitrule = "gini",
                     .min.node.size = 5)

# setting the control parameters
rfControl <- trainControl(
  method = "oob", number = 5,
  verboseIter = TRUE)

rf_model <- train(stroke ~ .,train_set,
                  method = "ranger",tuneLength = 3,
                  tuneGrid = rfGrid,trControl = rfControl)

# testing with balanced test set
confusionMatrix(predict(rf_model,test_set),test_set$stroke)
# Accuracy for the random forest with balance data
F1_Score(test_set$stroke,predict(rf_model,test_set))

# testing with original data
confusionMatrix(predict(rf_model, post_dataa) ,post_dataa$stroke)
# Accuracy for the random forest with original data
F1_Score(post_dataa$stroke,predict(rf_model,post_dataa))
