# Text Classification Case Study


## My Goal

The goal of this case study is to build a model that identifies which mobile phone apps are related to fitness.

## The Data

* One file contains text descriptions for 2500+ mobile phone apps
* One file contains the labels indicating if the app is a fitness app or not

## Results

Using an SGD classifier model, I was able to get an ROC AUC score of 97.6% on the test data. (This means that given one true positive and one true negative, my model would correctly predict that the fitness app is more likely to be a fitness app compared to a not-fitness app about 97.6% of the time.)

The top 20 word-roots in the descriptions that indicated a fitness-related app were (in order):

 - workout
 - fit
 - exercis
 - health
 - class
 - train
 - bodi
 - run
 - calori
 - daili
 - ab
 - diet
 - muscl
 - gym
 - track
 - activ
 - record
 - mind
 - program
 - paleo

For the most part, these words make sense as the top fitness-related words.


## My Approach


**First, I explored the text.**
 - There seemed to be no punctuation in the original text, so I didn't need to worry about removing it.
 - Most of the text seemed to be in English, although not all. The non-english descriptions were in a variety of other languages.
 - The number of words in the text description per app was about 125 words, but varied between a minimum of 4 words and a maximum of 500 words with a standard deviation of 94 words. That tells me that the number of words in each text description is highly variable.


 **Based on my initial exploration, I chose to use the following initial models featurization techniques:**
  - *Snowball Stemming*. I chose to start with a stemmer to get words to their root form but not a lemmer, and see what performance that would get. (In practice, I've often seen lemmers not add a ton of performance). Also, an English lemmer wouldn't work for any non-english text, however, at least for other romance languages, an english stemmer may have some impact. I chose the Snowball Stemmer as a solid middle-of-the-road start.
  - *tf-idf*: Because there is high variance in the number of words per app description, I chose to use a term frequency counter instead of bag of words. If an app description has only 4 words and one of them is "health", that is more significant than an app description that has 200 words and one of them is a variant of "health".
  - *Multinomial Naive Bayes*: This in an often used model for text classification, and often works with tf-idf.


**Validation and testing methodology:**
 - I set aside 30% of my data as my final test set. With the remaining 70% as my training data, I used cross-validation to estimate final performance of the models I experimented with and ultimately select a final model. This final model I then trained with my entire data set, and evaluated against my test data.


 **Featurization and Model Testing and Selection:**
 After running an initial model to get a starting ROC AUC score, I:
 - Compared TDIDF to bag of words approach, TFIDF did perform better.
 - Compared just a stemmer vs. just a lemmer vs using both. Just the stemmer performed the best.
 - Compared different stemmers. Porter did about the same, I chose to keep Snowball.
 - Compared different models, including logistic regression and SGD classifiers with varied loss functions. The SGD classifier with a 'log' loss function performed the best.
