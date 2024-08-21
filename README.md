# Intro

I want to practice and improve my Data Science and Machine Learning skills. 

I took this list as a [starting point.](https://medium.com/coders-camp/180-data-science-and-machine-learning-projects-with-python-6191bc7b9db9) But I started doing ML and Data Science Videos I found intersting also in Youtube. 

# List 

1. [Stock Price Prediction & Forecasting with LSTM Neural Networks in Python](https://www.youtube.com/watch?v=CbTU92pbDKw) available as AXON_Stock_Forecasting_with_LSTMs.ipynb
2. [House Price Prediction in Python - Full Machine Learning Project](https://www.youtube.com/watch?v=Wqmtf9SA_kk) available as House_Price_Prediction.ipynb
3. [Income Prediction Machine Learning Project in Python](https://www.youtube.com/watch?v=dhoKFqhVJu0) based on the [Adult income dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset) available at Kaggle. **What was the objective?** To know what are the variales that determine the most the income of an adult. **What did I practice?** I practiced One Hot Encoding for creating dummie variables. Then I did a correlation table to see the variables that are more correlated to income. In this part I also made a zoom to stay with the features that have a higher correlation with income (above 80%). I split the data into ntrain and tes (I choose the 20% test size). I then did a Random Forest Classifier and finished with a score of 84%. I then performed a Hyperparameter tuning to find the model that fits the most my data. I used different n_estimators, max_depth, min_samples_split and max_features based on the [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).After tune modeling the score of the model was 86%. **Where to find my code?** [Check this link.](https://github.com/rulocastellanos/practice_data_science_ml/blob/main/Adult_income.ipynb)
4. [Flight Price Prediction in Python](https://www.youtube.com/watch?v=EMoLTicNR6w) **What was the objective?** The objective of the study is to analyse the flight booking dataset obtained from “Ease My Trip” website and to conduct various statistical hypothesis tests in order to get meaningful information from it. **What did I practice?** Preprocesing, creating one hot encoding, creating a Random Forest Model, creating predictions based on test, Creating ussing mathplotlib a regression line. **Where to find my code?** [Check this link.](https://github.com/rulocastellanos/practice_data_science_ml/blob/main/Flight_Price_Prediction.ipynb)
5. [Automated Text Summarization Using NLP](https://www.youtube.com/watch?v=NabFGP4yLnk&list=PL_1pt6K-CLoCM3kyfOfsX5vn-_U8w9b4H) **What is the objective?** Summarazinf the key ideas of a text using 2 techniques: the extractive text and abstractive text summarization. **What did I practice?** Tokenize the words removing stopwords using 2 different techniques. Calculate the frecuency of each word. Normalizing the frequency oof each word. Creating a score for each sentence and Summarization using transformers.**Where to find my code?** [Check this link.](https://github.com/rulocastellanos/practice_data_science_ml/blob/main/Automated_Text_Summarization_Using_NLP.ipynb)
6. [Loan Status Prediction](https://www.youtube.com/watch?v=p3-7qW_t5bw) **What is the objective?** Use diffeent ML algorithms to determine which one is better to determine if a person gets or no a loan. Originally the dataset included categorical and numerical feautures that could influence if a person has a loan (married, gender, education, income) **What did I practice?** I practiced the fact that ML datasets need to deal with the missing values, so I got rid of them either by dropping NA when the value is less than 5% of the observations or replecing them with the mode when it was greater than 5%. Then, using map function I transformed the categorical data to all numerical. Then I performed feature sscaling for the data to be in the same numeric range (betweeen 1, 0). Then I obtained the score of ML models of 5 different models: Linear regression, SVM, Decission Tree Classifier, Random Forest Classifier, Gradient Boosting Classifier. I created a dictionary to sstore the score of each one. Finally I used Hyper parameter tunning for improving the SVM and Random Forest Classifier. These both models obtained the highest score. At the end, using the joblib package I classify a new dataset whith the most tunned model.
7. [Time Series Forecasting with XGBoost](https://www.youtube.com/watch?v=vV12dGe_Fho) **What is the objective?** Create a model that predicts future foorectasting using XGBoost. 'time series forecasting example in python using a machine learning model XGBoost to predict energy consumption with python.' **What did I practice?** Setting the datetime as an index. Using seaborn package  to create time sseries graphs. Crate train and test based on arbitrary point in time. Create new date variables based on date (for example, month, day of year). see the importance of each feauture compared to hour target (consumption). Create an XGBoost model and modifing learning_rate = 0.01 to prevent overfitting. CReate a dataframe baased on the importance of the feature according to the model. Calculate the prediction and thhen the error comparing our prediction with our real data. **Where to find my code?** [Check this link. ](https://github.com/rulocastellanos/practice_data_science_ml/blob/main/Time_Series_Forecasting_with_XGBoost.ipynb)



# Possible next videos

1. https://youtu.be/VpMGXfhDQXc?si=JHGPGgpr-wL0n0-y
2. https://www.youtube.com/watch?v=cCONIdrM2VI&list=PL-Y17yukoyy0sT2hoSQxn1TdV0J7-MX4K
3. https://www.youtube.com/watch?v=hG8K5h2J-5g&list=PLdF3rLdF4ICQ4-fSEucMqoqMz1tEyjp9q
4. https://www.youtube.com/watch?v=MoqgmWV1fm8&t=1s
5. https://www.youtube.com/watch?v=oKYxU8Kr900&list=PL_1pt6K-CLoCM3kyfOfsX5vn-_U8w9b4H&index=2
6. https://www.youtube.com/watch?v=xi0vhXFPegw
7. https://www.youtube.com/watch?v=epidA1fBFtI
8. https://www.youtube.com/watch?v=vSgJ3bOyE0w
9. https://www.youtube.com/watch?v=CrSC1ZA9j0M (html)
   




