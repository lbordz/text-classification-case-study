# Text Classification Case Study


## My Goal

The goal of this case study is to build a model that identifies which mobile phone apps are related to fitness.

## The Data

* one file contains text descriptions for 2500+ mobile phone apps
* one file contains the labels indicating if the app is a fitness app or not

## Results

My final model had a ROC AUC score of 97.5% on the test data. This means that given one true positive and one true negative, my model would correctly predict that the fitness-app is more likely to be a fitness app compared to the not-fitness app 96% of the time.

The top 20 word-roots in the descriptions that indicated a fitness-related app were (in order):

 - workout
 - exercis
 - fit
 - app
 - train
 - bodi (body, bodies, etc)
 - weight
 - track
 - use
 - calori
 - time
 - muscl
 - get
 - run
 - health
 - calcul
 - help
 - program
 - diet
 - yoga
 - applic
 - daili
 - fat
 - food
 - activ

 For the most part, these words make sense as the top words.



## My Approach (Decisions I Made)


To approach this problem:

**First, I explored the text.**
 - There seemed to be no punctuation in the original text, so I didn't need to worry about removing it.
 - Most of the text seemed to be in English, although not all. The non-english descriptions were in a variety of other languages.
 - The number of words in the text description per app was about 125 words, but varied between a minimum of 4 words and a maximum of 500 words with a standard deviation of 94 words. That's a pretty large range, and tells me the number of words in each text was highly variable .


 **Based on my initial exploration, I chose to use the following initial models featurization techniques:**
  - *Snowball Stemming*. I chose to start with a stemmer to get words to their root form but not a lemmer, and see what performance that would get. (In practice, I've often seen lemmers not add a ton of performance). Also, an English lemmer wouldn't work for any non-english text, however, at least for other romance languages, an english stemmer may have some impact. I chose the Snowball Stemmer as a solid middle-of-the-road start.
  - *tf-idf*: Because there is high variance in the number of words per app description, I chose to use a term frequency counter instead of bag of words. If an app description has only 4 words and one of them is "health", that is more significant than an app description that has 200 words and one of them is a variant of "health".
  - *Multinomial Naive Bayes*: This in an often-used model for text classification, and often words with tf-idf.


**Validation and testing methodology:**

 - I set aside 30% of my data as my final test set. With the remaining 70% as my training data, I used cross-validation to estimate final performance of the models I experimented with and ultimately select a final model. This final model I then trained with my entire data set, and set to expected final performance of my model.


 **Model / Featurization Testing and Selection:**
 - I compared TDIDF to Back of words, TFIDF did perform better based on cross-validation with my test set.
 - I compared different stemmers. XX DID PERFORM BETTER***






....
