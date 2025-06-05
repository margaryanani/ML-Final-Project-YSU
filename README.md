# Credit Default Analysis
For this project I used a dataset that contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005. This is a classification problem where we predict if a client will default on their credit card payment.

You can find the code in the "Code" folder: 'credit_default_analysis.ipynb'
## Libraries
`import pandas as pd`

`import numpy as np`

`import matplotlib.pyplot as plt`

`import seaborn as sns`

`from sklearn.model_selection import train_test_split, cross_val_score`

`from sklearn.preprocessing import StandardScaler`

`from sklearn.linear_model import LogisticRegression`

`from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score`

`from sklearn.svm import SVC`

`from sklearn.neighbors import KNeighborsClassifier`

`from sklearn.tree import DecisionTreeClassifier`

`from sklearn.naive_bayes import GaussianNB`

`from sklearn.metrics import roc_curve, roc_auc_score`

`from imblearn.over_sampling import RandomOverSampler`

`from sklearn.ensemble import BaggingClassifier`

## Dataset Overview
Rows: 30,000, Columns: 25
- **ID**: ID of each client
- **LIMIT_BAL**: Amount of given credit in NT dollars (includes individual and family/supplementary credit
- **SEX**: Gender (1=male, 2=female)
- **EDUCATION**: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
- **MARRIAGE**: Marital status (1=married, 2=single, 3=others)
- **AGE**: Age in years
- **PAY_0**: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
- **PAY_2**: Repayment status in August, 2005 (scale same as above)
- **PAY_3**: Repayment status in July, 2005 (scale same as above)
- **PAY_4**: Repayment status in June, 2005 (scale same as above)
- **PAY_5**: Repayment status in May, 2005 (scale same as above)
- **PAY_6**: Repayment status in April, 2005 (scale same as above)
- **BILL_AMT1**: Amount of bill statement in September, 2005 (NT dollar)
- **BILL_AMT2**: Amount of bill statement in August, 2005 (NT dollar)
- **BILL_AMT3**: Amount of bill statement in July, 2005 (NT dollar)
- **BILL_AMT4**: Amount of bill statement in June, 2005 (NT dollar)
- **BILL_AMT5**: Amount of bill statement in May, 2005 (NT dollar)
- **BILL_AMT6**: Amount of bill statement in April, 2005 (NT dollar)
- **PAY_AMT1**: Amount of previous payment in September, 2005 (NT dollar)
- **PAY_AMT2**: Amount of previous payment in August, 2005 (NT dollar)
- **PAY_AMT3**: Amount of previous payment in July, 2005 (NT dollar)
- **PAY_AMT4**: Amount of previous payment in June, 2005 (NT dollar)
- **PAY_AMT5**: Amount of previous payment in May, 2005 (NT dollar)
- **PAY_AMT6**: Amount of previous payment in April, 2005 (NT dollar)
- **default.payment.next.month**: Default payment (1=yes, 0=no)

The dataset has no missing values, 13 float and 12 integer features.

### Class distribution 

No Default: 23,364 (≈ 77.88%)

Default: 6,636 (≈ 22.12%)

The dataset is imbalanced.

### Features
- Credit limit (LIMIT_BAL) ranges from 10,000 to 1,000,000.
- Age ranges from 21 to 79 years.
- Payment history variables (PAY_0 to PAY_6) range from -2 to 8.
- PAY_0 to PAY_6 are strongly correlated with each other and BILL_AMT1 to BILL_AMT6 as well.
- The target variable has the strongest correlation with PAY_0(0.32),PAY_2(0.26),PAY_3(0.24), PAY_4(0.24).
- Clients who defaulted had slightly lower credit limits on average. There are significant outliers in both groups.
- Default rates are more common in the younger age groups. As age increases, the default rate drops.
- There’s not much of difference in default rate between genders.

## Data Preprocessing
First I converted categorical variables in a DataFrame into a format suitable for machine learning models by transforming them into one-hot encoded columns. It replaces the original categorical columns with multiple binary columns that indicate the presence of each category. Then i removed unnecessary variables like `ID`, splitted the data into features (X) and target (y) and Scaled the features using StandardScaler.

## Modeling

For modeling I Train-Test Splited the data and since the dataset was imbalanced I used `RandomOverSampler` to balance the class distribution.

Then i trained 6 models: Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree, Naive Bayes, Bagging (Decision Tree).

After training I tested the models and calculated evaluation metrics: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`.

| Metric         | Logistic Regression | K-Nearest Neighbors | Support Vector Machine | Decision Tree | Naive Bayes | Bagging (Decision Tree) |
|----------------|---------------------|----------------------|-------------------------|----------------|--------------|--------------------------|
| CV Accuracy    | 0.674               | 0.684                | 0.722                   | 0.877          | 0.524        | 0.929                    |
| Test Accuracy  | 0.681               | 0.776                | 0.778                   | 0.734          | 0.262        | 0.806                    |
| Precision      | 0.363               | 0.481                | 0.485                   | 0.385          | 0.223        | 0.565                    |
| Recall         | 0.639               | 0.491                | 0.573                   | 0.394          | 0.978        | 0.434                    |
| F1 Score       | 0.463               | 0.486                | 0.526                   | 0.389          | 0.363        | 0.491                    |
| ROC-AUC        | 0.717               | 0.736                | 0.752                   | 0.611          | 0.729        | 0.746                    |

The best accuracy got Bagging, SVM and K-Nearest.

Best at Finding Defaults (Recall) Naive Bayes was by far the best at catching actual defaults (93% recall), Logistic Regression caught 66% of defaults.Other models struggled to find defaults.

when it came to Precise Predictions Bagging was most accurate (59% precision),SVM and K-NN had around 50% precision, Naive Bayes had very poor precision at only 25%.

Decision Tree showed a huge drop from 87.3% cross-validation accuracy to 73.5% test accuracy, which indicates overfitting. Naive Bayes assumes that all features are independent, which probably isn't true for credit card data. For example, a person's income and spending patterns are likely related, but Naive Bayes treats them as completely separate, thats why its precision is 25%.

Based on the results, I would choose Support Vector Machine (SVM) as the overall best model for this credit card default prediction problem, because it gave the most balanced performance, is consistent and reliable. 

