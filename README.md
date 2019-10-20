# IEEE-CIS Fraud Detection

# My Result

![alt text](https://github.com/MuhammedBuyukkinaci/IEEE-CIS-Fraud-Detection/blob/master/my_result.png)

# Lessons Learned

1) Create UID(Unique Identification) with card1,addr1 and D1. D1 is difference between when dard was published and when the cars was used. Create a new feature D1n, which is TransactionDay minus D1.

2) Postprocessing = Taking all predictions from a single client (credit card) and replacing them with that client's average prediction were useful.

3) Grouping columns with respect to their null counts is crucial.

4) Dimensionality reduction techniques for V features(340 features and 15 groups):
- Applying PCA on each group individually
- Selecting a maximum sized subset of uncorrelated columns from each group
- Replacing the entire group with all columns averaged. 
- https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id

5) Time Consistency Feature Selection: Train a model with ONE feature on first month. Predict the score of last month(validation data). If AUC is lower than 0.5 for validation data, think to drop it.

6) Adversarial Validation: Label your train data as 0(Don't forget to drop target columns for columns number consistency) and label your test data as 1. Build a ML model on merged data (train + test). If AUC is close to 0.5, it can be deducted that train and test data have similar patterns.

7) Permutation Importance: Fit a ML model on train data and predict on validation data and save the result. Then, shuffle each column of validation data one by one and save the results. If the shuffled performance is lower for a feature, we can say that that feature is important or vice versa.
https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html

8) Different Validation strategies: train on 2 months, skip 2 months and predict 2 months; train on 4 months, skip 1 month and predict 1 month; train on 5 months and predict 1 month.

9) train/test distribution consistency: Comparing histogram of a feature in train dataset with its distribution on test dataset. 
Use from scipy.stats import ks_2samp.

10) Client Consistency: Split train dataset by user_id. Put 80% of userid's in training and 20% in validation. There should be no overlapping. This approach checks how features perform on undiscovered clients.

11) Boosted trees(LGB,XGB,Catboost) have different feature important metrics.Split refers to quantity and gain refers to quality.

12) Transforming time delta features(D columns) to their fixed points which are in the past is crucial. Example : D15n = Transaction_Day - D15

13) Replacing the test predictions with target values in train is a post processing procedure that can boost our score.

14) Setting colsample_bytree = 0.5 in lightgbm can be useful while extracting features.

15) Dynamic Variables:Adding the mean of TransactionAmt to each transaction. For example, if a customer made 5 submissions with [10,20,30,60,40], its mean transactions are as follows: [10,15,20,30,32].

16) Adding a new column which indicates a values is np.nan or not can be very useful to train a NN.

17) Before training a NN, applying a log transformation before standardization can be implemented.

18) Decreasing the cardinality of categorical columns and then applying one-hot encoding can be applicable while training a NN.

19) Embedding Layer can be concatenated with the last hidden layer of neural network. This means that neural networks can be fed by two different-positioned input vectors.
https://deepctr-doc.readthedocs.io/en/latest/Features.html#deepfm

20) Bad guys stole credit cards. They tried more transactions with ONE card in a lone period or they tried more transactions with MANY cards in a short period.

21) UID = Unique Identification. UID can be done via card-based or user-based.

22) Target encoding should be made in out of fold validation set. This means that if we have 5 fold validation, the target endoded values of one fold should be calculated from 5-1=4 folds.

23) SEQ2DOC: Encoding in a way that giving more importance to last transactions and less importances to previous transaction. If a client has the following 5 isFraud from most recent to last recent: 1, 0, 0, 1, 1. Then you encode this sequence as 1.0011

