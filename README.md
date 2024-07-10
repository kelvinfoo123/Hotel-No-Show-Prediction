# Hotel-No-Show-Prediction
Objective: Build a machine learning model to predict when reservations will lead to a no-show cancellation. 

#### **Workflow**
<img width="1038" alt="Screenshot 2024-07-10 at 12 26 34" src="https://github.com/kelvinfoo123/Hotel-No-Show-Prediction/assets/112041340/c4d4de5f-fa40-4329-abe1-391478a89c79">

1. Dealing with Duplicate Records
- Dropping duplicate records.
  
2. Dealing with Missing Records
- Impute room type based on price and branch (eg. A room in Changi whose price <= 600 is classfied a single room).
- Impute prices using random forest regressor based on branch, booking month, difference between booking and arrival month, number of occupants, platform and room type (random forest regressor had a RMSE of 74.37). 

3. Feature Engineering
- Conversion of prices in USD to SGD.
- Lower case strings for arrival month, checkout month and booking month.
- Calculate difference between booking month and arrival month.
- Sum number of adults and number of children to be total number of occupants.
- Computed stay duration based on arrival date and checkout date.

4. Data Pre-processing
- One-hot encoding applied to categorical features.
- Standard scaling applied to numerical features.

5. Machine Learning
- Experimented with logistic regression, K-nearest neighbours, support vector machine, random forest classifier and XGBoost.
- Measured performance of models using recall, confusion matrix and ROC-AUC.
- Without tuning, random forest classifier performed the best (recall = 0.48).
- Optuna was used to tune the hyperparameters of random forest classifier and it boosted the recall significantly to 0.61.

**Chosen model: tuned random forest classifier**
