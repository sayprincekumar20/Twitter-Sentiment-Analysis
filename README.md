# Twitter-Sentiment-Analysis

 we try to implement an NLP Twitter sentiment analysis model that helps to overcome the challenges of sentiment classification of tweets. We will be classifying the tweets into positive or negative sentiments. The necessary details regarding the dataset involving the Twitter sentiment analysis project are:

The dataset provided is the Sentiment140 Dataset which consists of 1,600,000 tweets that have been extracted using the Twitter API. The various columns present in this Twitter data are:
* target: the polarity of the tweet (positive or negative)
* ids: Unique id of the tweet
* date: the date of the tweet
* flag: It refers to the query. If no such query exists, then it is NO QUERY.
* user: It refers to the name of the user that tweeted
* text: It refers to the text of the tweet

#  Twitter Sentiment Analysis Dataset: Project Pipeline
The various steps involved in the `Machine Learning Pipeline` are:

* Import Necessary Dependencies
* Read and Load the Dataset
* Exploratory Data Analysis
* Data Visualization of Target Variables
* Data Preprocessing
* Splitting our data into Train and Test sets.
* Transforming Dataset using TF-IDF Vectorizer
* Function for Model Evaluation
* Model Building
* Model Evaluation

#### Installation
To run this project, you need to have the following libraries installed:

bash
Copy code
*     pip install pandas numpy matplotlib seaborn scikit-learn
#### Data Exploration
We began the analysis with Exploratory Data Analysis (EDA) to understand the dataset. Key steps included:

* Data Cleaning: Removing duplicates, null values, and irrelevant information.
* Text Preprocessing: Tokenization, stemming, and lemmatization of tweets.
* Visualization: Word frequency graphs to identify the most common terms in positive and negative tweets.

### Model Comparison
Three different models were compared for their performance on the sentiment analysis task:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Bernoulli Naive Bayes
   The idea behind choosing these models is that we want to try all the classifiers on the dataset ranging from simple ones to complex models, and then try to find out the one which gives the best performance among them

Each model was evaluated using:

* F1 Score: To measure the model's accuracy in classifying tweets.
* Accuracy: The proportion of true results (both true positives and true negatives) among the total number of cases examined.
* Confusion Matrix: To visualize the performance of the model.
* ROC-AUC Curve: To assess the models' ability to distinguish between classes.

###  ROC-AUC Curve
The ROC-AUC curves for all three models were plotted to visualize their performance.

Results
After comparing the models, Logistic Regression was found to be the best fit for the dataset, achieving the highest F1 score, accuracy, and AUC.

* Logistic Regression F1 Score(0): 0.79
* Logistic Regression F1 Score(1): 0.78
* Logistic Regression Accuracy: 0.78
* ROC Curve (area= 0.78)
  
 Logistic Regression Confusion Matrix:


   *       [[40.21%, 9.66%],
           [12.14% , 37.99%]]

       
* SVM F1 Score(1): 0.79
* SVM F1 Score (0) : 0.80
* SVM Accuracy: 0.80
* ROC Curve (area= 0.80)

SVM Confusion Matrix:


*      [[39.49%, 10.38%],
       [10.07%, 40.06%]]

   
* Bernoulli Naive Bayes F1 Score (1) : 80.0
* Bernoulli Naive Bayes F1 Score: 80.0
* Bernoulli Naive Bayes Accuracy: 80.0
* ROC Curve (area= 0.80)

  
 Bernoulli Naive Bayes Confusion Matrix:


*      [[39.62%, 10.25%],
      [9.50%, 40.63% ]]

### Conclusion
The Logistic Regression model outperformed the other models in terms of accuracy, F1 score, and AUC, making it the ideal choice for sentiment analysis in this case


### Usage
The trained Logistic Regression model has been saved for further predictions. You can load the model and make predictions on new tweet data using the following code snippet:

*      import pickle

      # Load the saved model
      with open('trained_model.sav', 'rb') as file:
      loaded_model = pickle.load(file)

      # Predict on new data
      prediction = loaded_model.predict(X_new)


###  License
This project is licensed under the MIT License.

### Acknowledgements
* scikit-learn for machine learning tools.
* Pandas for data manipulation.
* Matplotlib and Seaborn for data visualization.

