---
layout: post
title: Tweet Classification on the MediaEval Benchmarking Initiative for Multimedia Evaluation Data
subtitle: A theoritic critical analysis of prospective approaches
cover-img: /assets/img/twitter_article_cover.jpg
thumbnail-img: /assets/img/twit_thumb.png
share-img: /assets/img/twitter_article_cover.jpg
tags: [Deep Learning, Data Science]
---

This analysis is my attempt at theorizing a solution for a problem in a situation where the complete dataset is not available yet. This work can be used as an example to generate similar theoretical proposals, which can be beneficial in the early stages of data consulting or research projects, where confidentiality has not yet been established.

# Introduction and Data analysis

## Problem Description
The task at hand was to build binary classification models based on the given dichotomy of tweets that are categorized into ‘real’ and ‘fake’. Five complete approaches to achieve this are discussed and critically analyzed. Their strengths and weaknesses are compared, and the approaches are ranked based on this analysis. The dataset provided was The MediaEval 2015 ”verifying multimedia use” dataset which consisted of social media posts for which the social media identifiers were shared along with the tweet text and some additional characteristics of the post.

## Dataset Characteristics
1) Data Format: The data was provided in the form of a text file with the fields separated by tabs. Two separate text files were provided, one for training and the other for testing. The data was converted into a CSV file using Microsoft Excel.
2) Data size and Volume: The dataset consisted of 14,483 tuples and 7 attributes to define them. The size of the (train and test) data was 3.28 MB put together.

## Data Features and Quality
1) Raw Dataset sample: The sample is shown in Fig 1. It is to be noted that the aim was to classify using the text and tweet metadata. Ergo, the images that are included in the original dataset had not been provided for this work.
2) Grouping by class labels: There are three classes in the raw dataset, namely Fake, Humor and Real as shown in Fig. 2. The fake class consists of Reposts of real multimedia, such as real photos from the past re-posted as being associated to a current event, digitally manipulated multimedia, and synthetic multimedia, such as artworks or snapshots presented as real imagery. The humor labelled tweets may be considered as fake, but it is better to discard them to avoid exacerbating the high data bias due to the existence of significantly higher number of fake tweets than that of real tweets (9464 fake tweets with humor included and 6833 fake tweets without inclusion of the same as compared to 5004 real tweets as depicted in Fig. 3).
3) Language distribution of tweets: As seen in Fig. 3, 77% of the tweets were in English, followed by Spanish which is at 8%. Remaining tweets of 30 languages were lower than 2% each and add up to a little over 14%. The language of tweets was detected using the ‘langdetect’ library and the tweets were translated to English using the ‘googletrans’ library.
4) The column attributes in the dataset with unique and null value count: The userID and the username field represent the same detail. Hence the username attribute was removed to avoid redundancy and overfitting. The class labels, timestamps, usernames, userids and imageids were missing for 15 rows. As this is a very small number in comparison to 14,483 and comes up to about 0.1% of data, it was considered best to drop these rows.
5) Overall Quality Assessment: The data quality was assessed based on six parameters [1]. The data had very low null value rate, indicating Completeness. The dataset is from 2015 and hence might not be best suited for the classification of tweets from 2020, resulting in poor Timeliness. However, the aim of this coursework is to specifically use the mediaeval 2015 dataset to solve the problem at hand. Thus, the Validity of the data is on point. Just over 86% of the tweet texts are unique. Ergo, the Uniqueness of the data, despite not being ideal, is certainly high. Lastly, since the dataset was extracted using the twitter API, the Accuracy and Consistency of the dataset is outstanding. Putting it all together, four parameters were found to be very high and one was found to be high. Just one parameter needed improvement and hence the quality of data was stated to be high.
6) Computation speed needed: The data, being text-only data and having a small size of 3.28 MB, did not demand extreme computational power. Google Colabs, a completely free service, was used to analyze the data and use it for training and testing purposes. The notebook ran on AMD EPYC 7B12 processor and its Memory usage was less than 1 GB.

# Algorithm Design
This section describes the five algorithms in detail along with the justifications for choices wherever needed.
## Approach 1
This is a conventional approach that follows the
‘go-to’ process for tweet classification.
### Preprocessing: 
The stopwords were to be
removed from the ‘tweetText’ field to prevent the
models from being influenced by words that might
not add much value to the meaning of the document
[10]. This was done using the ‘nltk’ stopwords
list. Tokenization was performed following this,
which was done using the nltk tokenizer. Then,
Lemmatization was to be performed to get the
root words, after performing POS tagging. It is
to be noted that lemmatization is preferred over
stemming because the former is more powerful as
it performs morphological analysis of words rather
than merely cutting of the suffix [11].

###  Feature extraction: 
The “Bag of n-grams”
representation was used for feature extraction
as it is “simple to understand and implement
and has seen great success in problems such as
language modeling and document classification”
[12]. The ‘CountVectorizer’ library was to be
used to generate the representation. Just unigrams
were considered in this approach to keep the
computational burden to a minimum. The obtained
count matrix will then be normalized using TF-IDF
(Term Frequency – Inverse Document Frequency),
using ‘TfidfTransformer’ library. TF-IDF is used
to reduce the influence of tokens that occur very
frequently giving them less generalization power
and making them less informative.

###  Feature selection and dimensionality reduction: 
Firstly, this approach focuses only
of the classification of tweets. Hence only the
Word vectors and the class label features shall
be taken. To reduce the dimensionality, Chi
squared test, being comparatively simpler and a
widely used test for statistical difference, was
performed to determine if a particular feature
and the class label were uniform. Features over
the threshold were kept and the rest were discarded.

### Modelling: 
Multinomial NB was suitable for
this approach as it was specifically made for classification with discrete features. It used abundantly
with word counts for text classification as both
”training and testing complexities are linear for
scanning” [2]. SVM also works great with text token
classification [16]. Hence, both models are trained
and the model with the higher F1-score shall be
used

## Approach 2
The second approach is based on a unique
two-level classification model [3]. The working is
portrayed in Fig. 4.


![image](https://user-images.githubusercontent.com/26760537/167206370-38e20f3b-1c23-4987-b27d-9a1cb578ba1c.png)


###  Preprocessing: 
The stop words were removed from the tweets using nltk stop words list. Following this, the tweets were tokenized, parts of speech were tagged and lemmatization was performed similar to the first approach.

###  Feature extraction: 
The tokens were then converted to feature vectors using the ‘TfidfVectorizer’ library. The ‘datetime’ python library was used to convert the given timestamps into Unix timestamps (number of seconds elapsed from 1970). This was done because this makes the timestamp into an ordinal attribute and hence can be directly fed to a model. The imageIDs were one hot encoded. One-hot encoding was chosen over merely assigning numbers as there is no preference order for the images. Thus, one hot encoding the images gives them equal weightage.

###  Feature selection and dimensionality reduction: 
The userID, tweetId, processed timestamp, one-hot encoded imageId and tokens are selected. Dimensionality reduction is omitted in this approach as it is a two-level approach and requires every bit of data available. The inter-relations between the tweets might be lost if the dimensionality is reduced.

###  Modelling:
The TFIDF vectors will undergo clustering. The clustering algorithm used was Kmeans clustering and the distance measure was
Jaccard’s distance as the order of words in the tweets
was not considered. These choices were made as
this combination was found to work well in similar
works [4,5]. After obtaining the right number of
clusters experimentally by checking performance
of the final results of the approach, the obtained
groups were labelled as topics. All the tweets were
classified as real or fake using an SVM. The topic
was labelled real if more than half the tweets
belonging to the topic had the class label real.
Then, the results of this topic level classification
were added to each tweet in the dataset. Now,
A random forest classifier was used for the final
tweet level classification. Random forest was chosen
because it reduces overfitting in decision trees while
improving the accuracy [17] and is also great with
high dimensional data [18]. ‘scikit-learn’ machine
learning library was used for the SVM and random
forest algorithms

## Approach 3

###  Preprocessing:
The tweets must first be normalized by matching patterns using regular expressions. This was done to “map syntactic, lexical and Twitter specific forms found in tweets to their normalized forms” [6]. Hashtags and mentions are Twitter specific forms, and the rest are lexical forms. These were stored as separate attributes. The tweets were then tokenized, and the stop words were removed like in approach 1. Moreover, there are just 379 unique ‘imageId’ values out of 14,468 rows. Hence, this attribute could greatly help with distinguishing power and consequently was included.

###  Feature extraction and Selection:
The text tokens, the engineered hashtags and mentions fields, imageId and the class labels were selected. The text tokens were transformed into their respective TF-IDF representations. The imageId, hashtags and mentions are one -hot encoded using the ‘sklearn’ one-hot encoder library. Dimensionality Reduction was not applied to the whole dataset to avoid loss of information implied by inter tweet relations via user mentions, hashtags and imageIds. However, PCA was used to reduce the dimensionality of only the TF-IDF representations of the tokens. This ensures low redundancy and correlation amongst features through orthogonal components that have a high differentiating potential [14].

###  Modelling:
A deep neural network is best suited for this approach as the number of features is comparatively larger than the other approaches. Deep Neural networks consist of a fair number of neurons. Each neuron is a powerful classifier and hence a deep network of them will do a stellar job with a larger number of features when compared to conventional Machine Learning algorithms [7]. The architecture included an embedding layer, a flatten layer and two dense layers with ReLU and Sigmoid activation functions respectively. ReLU, being comparatively newer, is known to give better results than other functions and Sigmoid is used in the output layer as it is perfect for binary classification. Dropout regularization was also used to avoid overfitting. Adam Optimizer, which almost always outperforms other optimizers [19], was used and binary cross entropy was used as the loss function. This architecture was chosen as it performed well in a similar problem [13].

## Approach 4
This approach classifies tweets using a Recurrent
neural network (LSTM)

###  Preprocessing:
Special characters were
removed and then the words were tokenized by
converting each sequence into an integer encoded
representation and normalizing the length of the
sequences [8]. This specific approach is used as it
tailors the data for the LSTM. The real labels are
set to 1 and the fake ones are set to 0. There is
no stopword removal and TF-IDF is not used as
the LSTM takes the order of the words in tweets.
Hence the preprocessing is more superficial when
compared to other approaches.

###  Feature extraction and Selection:
The
tokenized tweets and the label column were
considered. Transfer Learning was used for
feature extraction. Glove embeddings and twitter
embeddings models were taken and used rather
than training the word vectors from scratch.
Both the embeddings were stacked together using
‘numpy.hstack’ method. The stacking demonstrated
better performance than using any one of the
embeddings or training them from scratch [8].

###  Modelling:
A bi-directional LSTM neural network was used for modelling for better contextual
learning. The network starts off with an Embedding
layer which was followed by dropout regularization.
Dropout was used instead of L2 regularization as it
strengthens the capability of individual neurons and
reduces dependence. Then a bi-directional LSTM
layer was used which was then followed by Convolutions (64,4) stacked on top of each state vector
in the LSTM. This is a common practice for the
classification of images and assumes that ”related
information is locally grouped together” [8]. Max
Pooling is applied after the convolutions before
moving any further. Then comes the Dense 64 layer
with ReLU activation function, which is a feed
forward network to interpret the output of the LSTM
layer. Finally, a Dense layer with Sigmoid activation
function acts as the output layer. The binary crossentropy loss function was used along with Adam
optimizer for evaluation. Adam optimizer, being the
best, is also best suited for LSTM networks [20].
Hyperparameter Tuning of Batch Size, Number of
epochs, Dropout Rate, LSTM Size was performed
to further enhance performance.

## Approach 5
This method works on the principle of ensemble
learning. It uses the algorithms of the approaches
1,2,3 and 4. This method has the potential to deliver
outstanding accuracy because the approaches used
focus on and cover different aspects. The first
approach focuses solely on the classification of
tweets. The second approach employs a unique
two-level classification approach which focuses
on the connections between tweets. The third is
a single level approach uses similar preprocessing
to the second approach, but it uses a deep neural
network to estimate unique hidden patterns. The
fourth approach uses an LSTM model with great
potential due to its reputation with textual data.
Due to this diversity of approaches, they seem to
complement each other and focus on what the other
algorithms overlooked. Ergo, when put together in
an ensemble voting scenario, they are guaranteed
to give a good result. Note that each approach is
made to focus on a vital aspect and hence, they are
all equally important. Hence voting is apposite to
this scenario.

###  Preprocessing, Feature extraction, Feature selection and Dimensionality reduction:
The four approaches stated in A,B,C and D were used as sub models of this ensemble model. The overview is portrayed in Fig. 5.

###  Modelling:
The predictions from all 4 models
were evaluated and the outcomes were recorded.
The class with the maximum number of votes was
taken as the outcome. Since there are four sub
models, there is a chance of a tie. In case of a tie, the
votes of approaches 2.1 and 2.2 shall be multiplied
by 0.5, that is they will be worth half their initial
value. This is due to the fact that the approaches in
2.3 and 2.4 have deep learning models and almost
always outperform the machine learning approaches
in the former two approaches and hence act as the
tie breakers.
![image](https://user-images.githubusercontent.com/26760537/167206101-c0ae6d8a-0e96-44cb-9d18-895584fdd53b.png)


# Evaluation
## Strengths and Weakness analysis of the approaches
### Approach 1
a) Strengths
i) The approach does not require heavy
processing power. Performing Chisquared test further takes this down
by removing features with high correlation.
ii) Lemmatization gets the root word
through morphological analysis. This
results uniform word vector representations with less noise.
iii) Multinomial NB and SVM are
known to be perfect for text tokens
classification in the literature [15].
b) Weaknesses
i) Technique does not preserve the order of the words.
ii) TF-IDF model is based on the bagof-words model and hence does not
capture the position in text, semantics, and co-occurrences amongst
documents. Therefore, TF-IDF is
only useful as a lexical level feature
[21].
iii) An ensemble of the two models
could further boost performance.

### Approach 2
a) Strengths
i) The two-level approach takes into
account the inter tweet relations.
ii) Attributes other than just the tweets
(userID, tweetId, timestamp, imageId) were considered resulting in
better performance on data that are
structurally similar to that in the
dataset.
iii) Clustering the tweets based on topic
helps uncover unique patterns in the
data that none of the other approaches consider.
b) Weaknesses
i) Determining the perfect number of
clusters is difficult.
ii) Two models without any dimensionality reduction demands powerful
hardware.
iii) The performance of the second
model could be severely affected if
the performance of the first level is
not up to the mark

### Approach 3
a) Strengths
i) Considers mentions and hashtags as
separate attributes to uncover patterns that are unique to this approach.
ii) Hyperparameter tuning performed
resulting in a near perfect model.
iii) Deep neural network used and hence
higher accuracy than other machine
learning functions is highly probable.
b) Weaknesses
i) High number of features might make
this approach susceptible to the curse
of dimensionality.
ii) The amount of data might not be sufficient for the neural network training
iii) Does not consider the order of the
words

### Approach 4
a) Strengths
i) Word context is considered by the
LSTM network.
ii) LSTM neural network with hyperparameter tuning guarantees great performance.
iii) Regularization is performed to avoid
overfitting.
b) Weaknesses
i) Being a deep learning approach, it is
computationally intensive.
ii) The dataset size might not be enough
for ideal results and the tweets might
be very small for the LSTM to draw
meaningful patterns from the order
of words.
iii) Convolutions stacked on top of each
state vector in the LSTM might end
up hurting the performance

### Approach 5
a) Strengths
i) Being an ensemble approach, the
best performance is highly likely.
ii) Uses sub models that focus on different aspects guaranteeing greater
depth in predictions.
iii) Equal voting is used to make the final classification consequently giving
equal weightage to all the models.
b) Weaknesses
i) Very heavy to train. Four full-fledged
models are used and hence even
the classification might not be quick
enough for real-time usage.
ii) The computational cost might not be
worth the boost in performance.
iii) The performance of the approach
could drastically plummet if the data
size is not sufficient for the deep
learning models because they act as
the tiebreakers.

## Ranking
The outcome of this ranking is to pick the
approach that is best potential approach for the
task at hand. Two main factors shall be considered
while ranking the approaches, namely speed and
correctness. Tweet classification is a task that needs
real time results due to the plethora of tweets every
second and the intention to quickly pick-up current
affairs information that is not fake. Correctness is
considered because the whole idea of classifying
tweets to gather information for decision making is
voided when a majority of tweets are misclassified.
The best approach for the problem at hand is
approach 3. It uses a deep neural network that is
known to outperform machine learning algorithms
almost always, given enough data. Even though the
training process is computationally intensive, the
classification process does not take much longer that
the other approaches. Moreover, Hyperparameter
tuning ensures that there is no overfitting despite
the high dimensionality. Ergo, this approach strikes
the right balance between accuracy and speed and
is ranked 1.
The next approach is the approach 4 as it uses an
LSTM network and has the potential of getting excellent results when compared to machine learning
approaches [9]. The sole reason for this approach
not being ranked first is that it does not consider any
other attributes other than the ‘tweettext’. Hence it
should perform better in more generalized situations
where only the tweet text is to be classified. This
approach cannot be guaranteed to outperform approach 3 on the media eval 2015 dataset and hence
is ranked lower.
Furthermore, approach 5 is ranked third. Despite the
fact that this approach is the most computationally
intensive, it can almost be guaranteed that this
approach will supersede the other models as it is
an amalgamation of all those models. Therefore, this
approach is perfect for situations where the accuracy
cannot be compromised.
Approach 2 is ranked 4. This is due to the reliability
of concept of the two-level classification. In the off
chance that the first level model performs poorly, it
can severely affect the performance of the second
model. Additionally, the model could be slower due
to having two sub models and consequently needing
2 runs. Despite these draw backs, this model has the
potential to outperform the higher ranked models if
implemented thoroughly.
Finally, Approach 1 is ranked fifth. The main advantage of this approach is that it is the fastest of the
bunch, but it does not consider any other attributes
other than the ‘tweet text’. The order of the words is
also not preserved. Hence, for low budget projects
with very dirty and non-uniform data from which
only the tweet text is a usable attribute, this method
will fit in perfectly. But due to these limitations,
high performance cannot be guaranteed for this
approach.
With respect to the wider literature, these are the
best algorithms when it comes to tweet classification
based only on the given Mediaeval 2015 dataset
without images. But, in the case of Mediaeval and
tweet classification in general, these approaches fall
short as they dont consider images. A few approaches such as the work by Christina Boididou et
al. supersede these approaches. Christina Boididou
et al. take into account forensic image data for more
effective classification [22].

# Conclusion
Five potential analogous approaches were discussed in this work. The preprocessing, feature extraction, feature selection and modelling steps were
described in detail. The justification for the design
choices was provided along with the approaches.
The strengths and pitfalls of all the approaches were
discussed and the approaches were ranked bases
on speed and correctness. The final order of the
approached in decreasing order was hypothesized
to be: Approach 3 > Approach 4 > Approach 5 >
Approach 2 > Approach 1. A few key takeaways are
that the Deep Learning approaches will supersede
the machine learning algorithms with sufficient data
and enough computational power. Regularization,
feature selection and dimensionality reduction are
to be carefully used to reduce overfitting and make
the models efficient, while keeping loss of data in
mind. Hyperparameter tuning must be performed
wherever possible to make the models appropriate
for the current scenario. The order of the tweets can
help uncover important information even though it
is often neglected in terms of tweets due to their
small size.
Despite the thorough analysis and the inclusion of
five complete approaches, there are a few things
were omitted and must be resolved in the future.
The images have not been considered and evaluated. Adding another level of image classification
along with this can give rise to more interesting
approaches and can significantly boost accuracy.
Cross-Validation must be performed and the F1
scores along with the time taken by the approaches
must be experimentally determined and be considered for the ranking

# References
1. IT Pro team: ’How to measure data quality’, 2 Mar
2020,https://www.itpro.co.uk/business-intelligence-bi/29773/howto-measure-data-quality
2. C.D. Manning, P. Raghavan and H. Schuetze
(2008). Introduction to Information Retrieval. Cambridge
University Press, pp. 234-265. https://nlp.stanford.edu/IRbook/html/htmledition/naive-bayes-text-classification-1.html
3. Zhiwei Jin, Juan Cao, Yazi Zhang, and Zhang Yongdong. 2015.
MCG-ICT at MediaEval 2015: Verifying Multimedia Use with a
Two-Level Classification Model Proceedings of the MediaEval
2015 Multimedia Benchmark Workshop.
4. Tripathy, R.M. , Sharma, S. , Joshi, S. , Mehta, S. and Bagchi,
A. (2014), “Theme based clustering of tweets”, Proceedings
of the 1st IKDD Conference on Data Sciences, ACM, March,
pp. 1-5.
5. ’Jaccard index’, Wikipedia:
https://en.wikipedia.org/wiki/Jaccard index
6. Nicolas Foucault and Antoine Courtin. 2016. Automatic
Classification of Tweets for Analyzing Communication Behavior
of Museums. In Proceedings of the Tenth International Conference
on Language Resources and Evaluation (LREC 2016), Nicoletta
Calzolari (Conference Chair), Khalid Choukri, Thierry Declerck,
Sara Goggi, Marko Grobelnik, Bente Maegaard, Joseph Mariani,
Helene Mazo, Asuncion Moreno, Jan Odijk, and Stelios Piperidis
(Eds.). European Language Resources Association (ELRA),
Paris, France
7. Faizan Shaikh, ’Deep Learning vs. Machine
Learning – the essential differences you need to
know!’, https://www.analyticsvidhya.com/blog/2017/04/comparisonbetween-deep-learning-machine-learning
8. Deep Learning(LSTM) for Tweet Classification:
www.kaggle.com/mkowoods/deep-learning-lstm-for-tweetclassification
9. Sahoo A.K., Pradhan C., Das H. (2020) Performance Evaluation
of Different Machine Learning Methods and Deep-Learning Based
Convolutional Neural Network for Health Decision Making. In:
Rout M., Rout J., Das H. (eds) Nature Inspired Computing for
Data Science. Studies in Computational Intelligence, vol 871.
Springer, Cham. https://doi.org/10.1007/978-3-030-33820-6 8
10. Shubham Singh, ”NLP Essentials: Removing Stopwords
and Performing Text Normalization using NLTK and spaCy in
Python”, https://www.analyticsvidhya.com/blog/2019/08/how-toremove-stopwords-text-normalization-nltk-spacy-gensim-python/
11. Hafsa Jabeen, ”Stemming and Lemmatization in Python”,
https://www.datacamp.com/community/tutorials/stemminglemmatization-python
12. Jason Brownlee, ”A Gentle Introduction to the Bag-of-Words
Model”, https://machinelearningmastery.com/gentle-introductionbag-words-model/
13. Nikolai Janakiev, ”Practical Text Classification With
Python and Keras”, https://realpython.com/python-keras-textclassification/
14. ”Principal Component Analysis”, International Encyclopedia
of Education (Third Edition), 2010
15. ”Choosing what kind of classifier to use”,
https://nlp.stanford.edu/IR-book/html/htmledition/choosingwhat-kind-of-classifier-to-use-1.html
16. Monkey Learn, ”Text Classification”,
https://monkeylearn.com/text-classification/
17. Great Learning Team, ”Random Forest Algorithm- An
Overview”, https://www.mygreatlearning.com/blog/randomforest-algorithm/
18. Julia Kho, ”Why Random Forest is My Favorite Machine
Learning Model”, https://towardsdatascience.com/why-randomforest-is-my-favorite-machine-learning-model-b97651fa3706
19. Sanket Doshi, ”Various Optimization Algorithms For Training
Neural Network”, https://towardsdatascience.com/optimizers-fortraining-neural-network-59450d71caf
20. Karsten Eckhardt, ”Choosing the right Hyperparameters for a simple LSTM using Keras”,
https://towardsdatascience.com/choosing-the-righthyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046
21. Prasoon Singh, ”Fundamentals of Bag Of Words and
TF-IDF”, https://medium.com/analytics-vidhya/fundamentals-ofbag-of-words-and-tf-idf-9846d301ff22
22. Christina Boididou, Katerina Andreadou, Symeon
Papadopoulos, Duc-Tien Dang-Nguyen, Giulia Boato, Michael
Riegler, Yiannis Kompatsiaris, et almbox.. 2015. Verifying
Multimedia Use at MediaEval 2015.. In MediaEval
