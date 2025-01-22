# Module-17-Practical-Assignment-3
This repo contains the files for the PCMLAI Module 17 Practical Assignment 3

## Overview
In this practical application, your goal is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.  We will utilize a dataset related to marketing bank products over the telephone.  

## Getting Started
Our dataset comes from the UCI Machine Learning repository [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing).  The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns.

### Problem 1: Understanding the Data

To gain a better understanding of the data, please read the information provided in the UCI link above, and examine the **Materials and Methods** section of the paper.  How many marketing campaigns does this data represent?

- The data from a Portuguese banking institute gives the results of direct marketing campaigns done by the company. These campaigns were done through phone calls. The phone calls involve telephone and cellular calls. In total, there were 41,188 calls. Of these calls, 15,044 were telephone calls and 26,144 were cellular calls. The highest number of times contact had to be made during the campaign was 56 times.

### Problem 2: Read in the Data

Use pandas to read in the dataset `bank-additional-full.csv` and assign to a meaningful variable name.

- It should be noted that the UCI Machine Learning repository provided multiple versions of this dataset which includes samples of the original dataset. For the purpose of this project, only the original full dataset was read in so that the data could personally and easily be split for training and testing.

  ### Problem 3: Understanding the Features


Examine the data description below, and determine if any of the features are missing values or need to be coerced to a different data type.


```
Input variables:
# bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
```

- There are no missing values from this dataset. However, it should be noted that this dataset is significantly unbalanced with about 88.73% of people choosing not to subscribe to the term deposit and only 11.27% of people choosing to subscribe. This dataset is filled with floats, integers, and objects. Some features may need to be converted/encoded later during feature selection before modeling. This dataset was filled with a few duplicates which were removed.

- During this time, some data analysis was reformed to identify some features that are likely to affect the target outcome to help with modeling and predicting. Through a histogram visualizing the job column of the dataset, it can be seen that there are various jobs that have a higher rate of choosing to subscribe to the term deposit. These jobs include admin, technician, and more.
![Number of accepted subscriptions by job](https://github.com/user-attachments/assets/9d9a8f88-8d17-4850-9c60-0f48cfdaec4d)

- During data analysis, the jobs admin, technician, student, and retired were chosen for comparison. Despite the fewer subscriptions, fewer retired people and students were contacted, making the percentage who subscribed within those specific groups high. The comparison was made between married individuals with a technician, admin, student, or retired job to all others. The results showed that of the married individuals with a technician, admin, student, or retired job, 14.47% said "yes" to subscribing while 10.83% of all others said "yes". This shows that these jobs and a married status are characteristics that are likely to lead a person to say "yes" to a subscription. It should be noted that the high "no" rate has been observed but will not be extremely important considering how significantly unbalanced the data is.
![Subscription Acceptance for Married People With a Technician, Admin, Student, or Retired  Job to All Others](https://github.com/user-attachments/assets/1106147f-2f40-445d-9f5b-078bc73ef8db)

- Another comparison performed was between those with a university degree and no personal loan and all others. These features were chosen as those with a university degree are likely to make more money than those without one. The lack of personal loans means that these individuals likely do not have any major debt, like a student loan, that would prevent them from saying "yes" to a subscription. The results of the analysis show that 13.98% of those with a university degree and no personal loan said "yes" to a subscription while 10.40% of all others said "yes".
![Subscription Acceptance for People With a University Degree and No Personal Loan to All Others](https://github.com/user-attachments/assets/bd4fd47e-b809-4baa-8cfb-7dec9e97a348)

### Problem 4: Understanding the Task

After examining the description and data, your goal now is to clearly state the *Business Objective* of the task.  State the objective below.

- A Portuguese banking institution is attempting to get more clients to subscribe to a term deposit. The institution is reaching out to various clients through phone calls. Unfortunately, individual phone calls must be made to various clients with each client likely requiring multiple calls. This is time-consuming and unrewarding, especially if the client refuses to subscribe to a term deposit in the end. Therefore, the Portuguese banking institution must figure out which characteristics describe a client that will likely say "yes" to subscribing to a term deposit. Using classification models to determine the accurate model with the best features, the institution will be able to direct its marketing campaigns to clients that are more likely to subscribe which will bring in more business to the institution. 

### Problem 5: Engineering Features

Now that you understand your business objective, we will build a basic model to get started.  Before we can do this, we must work to encode the data.  Using just the bank information features, prepare the features and target column for modeling with appropriate encoding and transformations.

- To prepare the features for modeling, encoding had to be done as many of the seemingly important features were nonnumerical. For starters, the "y" column which displayed the target variable of whether someone said "yes" or "no" to the subscription was encoded. "No" is represented by "0" and "yes" is represented by "1". Further visualization was done to identify features that would be encoded versus those that would be deleted. The categorical features that were visualized include the 'job', 'marital', 'education', 'default', 'housing', 'loan', 'month', and 'day_of_week' columns. These visualizations show how many more people above the mean choose to subscribe out of all the people within each category. For example, although more people with an admin job said "yes", these visualizations show that out of the total number of students asked, most of them said "yes" compared to other jobs. These plots can also help with feature selection as they contributed to choosing retired people and students for the previous data analysis.
![The acceptance of categorical features](https://github.com/user-attachments/assets/2fbfb67e-54b1-4f9d-b417-12487dc80982)

- The non-numerical features that went through label encoding include the 'job', 'marital', 'education', 'housing', 'loan', 'month', and 'day_of_week' features. Although not all of these features are used for modeling, they can be used for further/future modeling. The features that were removed include the 'default', 'contact', 'poutcome', and 'duration' features. This is because the 'default' column was significantly imbalanced with only 3 people with a defaulted loan; therefore, this column likely does not have an effect on modeling and predicting. 'Contact' was removed as it does not seem to be very important beyond data recording. The 'poutcome' column was deleted as most of the values were "nonexistent". Despite being deemed important, the 'duration' column was deleted as the duration of the call cannot be determined prior to the call; therefore, this column holds no value for predictive modeling. After encoding, the index was reset.

  ### Problem 6: Train/Test Split

With your data prepared, split it into a train and test set.

- The data was split with the test size being 20%, leaving 80% of the data for training. The features that will be used for basic modeling are the 'job' and 'marital' features. The "y" column listing if the person chose to subscribe or not is the target.

### Problem 7: A Baseline Model

Before we build our first model, we want to establish a baseline.  What is the baseline performance that our classifier should aim to beat?

- A random classifier model was used as the baseline. This model took about 0.0022 seconds to train. Recall will be the initial evaluation metric with a confusion matrix being used as well. The recall score of the random classifier model is about 0.7977. There were a significant number of true negatives and a low number of true positives. However, the number of false negatives and false positives were about the same and higher than the true positives. This is the baseline performance that we should aim to beat.
![Confusion matrix of the Baseline model](https://github.com/user-attachments/assets/72fab45e-61b1-439c-b73c-6ff9dce0b54d)

### Problem 8: A Simple Model

Use Logistic Regression to build a basic model on your data.  

- A logistic regression model was built within 0.0220 seconds.

### Problem 9: Score the Model

What is the accuracy of your model?

- The recall score of the logistic regression model was 0.8821. From the confusion matrix, there is a high number of true negatives with the rest going to the false negatives. There were no false positives or true positives.
![Confusion matrix of the Logistic Regression model](https://github.com/user-attachments/assets/72fdbcc8-09f6-458b-9008-4c326b833f32)

### Problem 10: Model Comparisons

Now, we aim to compare the performance of the Logistic Regression model to our KNN algorithm, Decision Tree, and SVM models.  Using the default settings for each of the models, fit and score each.  Also, be sure to compare the fit time of each of the models.  Present your findings in a `DataFrame` similar to that below:

| Model | Train Time | Train Accuracy | Test Accuracy |
| ----- | ---------- | -------------  | -----------   |
|     |    |     |     |

- A knn model was built within 0.0078 seconds. The recall score of the knn model was 0.7927. From the confusion matrix, there is a high number of true negatives and a low number of true positives. The second highest is the false positives and the second lowest is the false negatives.
![Confusion matrix of the KNN model](https://github.com/user-attachments/assets/365ffc79-b3a3-4af9-a2e1-3f28e08817ac)

- A decision tree model was built within 0.0064 seconds. The recall score of the decision tree model was 0.8821. From the confusion matrix, there is a high number of true negatives with the rest going to the false negatives. There were no false positives or true positives.
![Confusion matrix of the Decision Tree model](https://github.com/user-attachments/assets/20c32a96-72f6-4140-82d7-f49268ca5b63)

- A svm model was built within 1.9802 seconds. The recall score of the svm model was 0.8821. From the confusion matrix, there is a high number of true negatives with the rest going to the false negatives. There were no false positives or true positives.
![Confusion matrix of the SVM model](https://github.com/user-attachments/assets/3df3b3f0-3f0c-4973-85ad-316abad9366c)

- Below is the filled-in table from earlier. However, it should be noted that accuracy is only listed here because the table asks for accuracy. Accuracy is not the best performance metric for this data considering the data is so unbalanced. Therefore, despite this table, the performance of the models will be evaluated based on the previously mentioned recall scores and confusion matrices.
![Time and Accuracy](https://github.com/user-attachments/assets/ed1e7342-67fd-4e96-9d82-3980cd284555)

### Problem 11: Improving the Model

Now that we have some basic models on the board, we want to try to improve these.  Below, we list a few things to explore in this pursuit.

- More feature engineering and exploration.  For example, should we keep the gender feature?  Why or why not?
- Hyperparameter tuning and grid search.  All of our models have additional hyperparameters to tune and explore.  For example the number of neighbors in KNN or the maximum depth of a Decision Tree.  
- Adjust your performance metric

- Further feature engineering would definitely be helpful. In the full dataset that is observed, there is no 'gender' column already. However, this column shouldn't be added as gender is not a determining factor for whether a person has the means to subscribe to a deposit or not. However, other features could be used as only the 'job' and 'marital' features have been used for these models so far.
- Hyperparameter tuning and grid search can be done on each of the four modeling techniques, logistic regression, knn, decision trees, and svm. The performance metrics are also adjusted as grid search finds the "best" parameters for each model. For the logistic regression model, the best parameter out of all those tried was "1".
- Performing GridSearchCV on a KNN model is computationally expensive. Therefore, the hyperparameter 'n' was tuned. Setting the n value to 10 was tried. The knn with n=10 model was built within 0.0070 seconds. The recall score of the knn model with n=10 was 0.5. From the confusion matrix, there is a high number of true negatives with the rest going to the false negatives. There were no false positives or true positives.
![Confusion matrix of the KNN model with N=10](https://github.com/user-attachments/assets/f0aea927-035a-4926-9c62-533c8205c3a4)

- The maximum depth of the decision tree model is 8. This means that the maximum number of levels/splits for this model is 8 levels/splits. It should be noted that although a higher maximum depth captures more complex patterns in the data, it can lead to overfitting. Therefore, a decision tree with a max_depth of 4 was made. The decision tree with max_depth=4 model was built within 0.0051 seconds. The recall score of the decision tree model with max_depth=4 model was 0.5. From the confusion matrix, there is a high number of true negatives with the rest going to the false negatives. There were no false positives or true positives.
![Confusion matrix of the Decision Tree model with max_depth=4](https://github.com/user-attachments/assets/b31f1d07-0515-4e4c-9043-f19e6c994041)

- For the SVM model, GridSearch CV was not performed as doing so on SVM is too computationally expensive. Instead, a polynomial kernel was added to the SVM model. The svm model with a polynomial kernel was built within 2.5941 seconds. The recall score of the svm model with a polynomial kernel was 0.5. From the confusion matrix, there is a high number of true negatives with the rest going to the false negatives. There were no false positives or true positives.
![Confusion matrix of the SVM model with a polynomial kernel](https://github.com/user-attachments/assets/f1da179e-1e8e-4c0d-bfe6-fe6e13df08a1)

### Findings

As a reminder, the goal of creating these models is to help the Portuguese banking institution predict if a customer will subscribe to the term deposit or not. This will help the comany save time when it comes to their marketing campains as these campaigns involves indiviually calling customers multiple times and talking to them for various and long durations. Being able to know what features and characteristics customers are likely to high will help them target the campains to individuals who will likely subscribe. After some data analysis and predictive modeling, it seems that a decision tree model is the best for redicting if a customer will subscribe or not. The various factors that went into this result include the times, recall scores, and confusion matrices of the models. It should be noted that the main two models that performed well were the logistic regression and decision tree models. The knn and svm models were very computationally expensive. Although the knn model had a slightly higher recall score, the difference is not significant and the confusion matrix for this model shows many more false positives compared to other models. The knn model with n=10 showed a better confusion matrix than the knn model with n=1; however, this confusion matrix was no better than the other models. In addition to being computationally expensive, the svm model did not show better results when a polynomial kernel was added. Between a logistic regression model and a decision tree model, the decision tree model was less computationally expensive both with and without a specified maximun depth. Furthermore, decision trees work best for this dataset as they are better for imbalanced data and are not biased toward the majority class.

### Next Steps

Futher data collection, data cleaning, and predictive modeling is best. Other features that may affect a subscription decision should be considered like the number of children someone may have or the individuals income. Income is esecially important as the job feature seems to be effective during prediction. However, there are some jobs that may seem like they usually yield a high income when they actually don't. For example, someone who is self-employed or and entrprenuer could make a wide range of money that even various throughout the year significantly. Further data cleaning could be done to better encode the nonnumerical data. It should be noted that for the purpose of knowing exactly what label was encode to which feature, label encoding was done individually. Using the LabelEncoder() function would be faster in the future. Furthur feature selction could also be done. For the models created, features specific to the individuals who were called was used, like job and marital status. However, other features like the unemployment rate and month may be useful in predicting the best time to contact customers. Furthure paramteer tuning would also be useful. Consideirng the duration of the call was important yet was not useful for prediction, the Portuguse banking institution could also try other marketing campagns like in-person ones that help lesson duration and contact while increasing the reach. Personally talking to someone can hel when tryingg to convince them of something in comparison to talking on the phone.

The link to the coding for this project is here
