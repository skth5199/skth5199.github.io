---
layout: post
title: Tweet Classification on the MediaEval Benchmarking Initiative for Multimedia Evaluation Data
subtitle: A Critical Analysis of five approaches
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [Deep Learning, Data Science]
---

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
1) Preprocessing: The stopwords were to be
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
2) Feature extraction: The “Bag of n-grams”
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
3) Feature selection and dimensionality
reduction: Firstly, this approach focuses only
of the classification of tweets. Hence only the
Word vectors and the class label features shall
be taken. To reduce the dimensionality, Chi
squared test, being comparatively simpler and a
widely used test for statistical difference, was
performed to determine if a particular feature
and the class label were uniform. Features over
the threshold were kept and the rest were discarded.
4) Modelling: Multinomial NB was suitable for
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
1) Preprocessing: The stop words were removed
from the tweets using nltk stop words list. Following
this, the tweets were tokenized, parts of speech
were tagged and lemmatization was performed
similar to the first approach.
2) Feature extraction: The tokens were
then converted to feature vectors using the
‘TfidfVectorizer’ library. The ‘datetime’ python
library was used to convert the given timestamps
into Unix timestamps (number of seconds elapsed
from 1970). This was done because this makes the
timestamp into an ordinal attribute and hence can
be directly fed to a model. The imageIDs were one
hot encoded. One-hot encoding was chosen over
merely assigning numbers as there is no preference
order for the images. Thus, one hot encoding the
images gives them equal weightage.
3) Feature selection and dimensionality
reduction: The userID, tweetId, processed
timestamp, one-hot encoded imageId and tokens
are selected. Dimensionality reduction is omitted
in this approach as it is a two-level approach
and requires every bit of data available. The
inter-relations between the tweets might be lost if
the dimensionality is reduced.
4) Modelling: The TFIDF vectors will undergo
clustering. The clustering algorithm used was Kmeans clustering and the distance measure was
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
1) Preprocessing: The tweets must first be
normalized by matching patterns using regular
expressions. This was done to “map syntactic,
lexical and Twitter specific forms found in tweets
to their normalized forms” [6]. Hashtags and
mentions are Twitter specific forms, and the rest
are lexical forms. These were stored as separate
attributes. The tweets were then tokenized, and
the stop words were removed like in approach
1. Moreover, there are just 379 unique ‘imageId’
values out of 14,468 rows. Hence, this attribute
could greatly help with distinguishing power and
consequently was included.
2) Feature extraction and Selection: The text
tokens, the engineered hashtags and mentions fields,
imageId and the class labels were selected. The
text tokens were transformed into their respective
TF-IDF representations. The imageId, hashtags and
mentions are one -hot encoded using the ‘sklearn’
one-hot encoder library. Dimensionality Reduction
was not applied to the whole dataset to avoid loss
of information implied by inter tweet relations via
user mentions, hashtags and imageIds. However,
PCA was used to reduce the dimensionality of
only the TF-IDF representations of the tokens. This
ensures low redundancy and correlation amongst
features through orthogonal components that have
a high differentiating potential [14].
3) Modelling: A deep neural network is best
suited for this approach as the number of features
is comparatively larger than the other approaches.
Deep Neural networks consist of a fair number of
neurons. Each neuron is a powerful classifier and
hence a deep network of them will do a stellar job
with a larger number of features when compared
to conventional Machine Learning algorithms [7].
The architecture included an embedding layer, a
flatten layer and two dense layers with ReLU and
Sigmoid activation functions respectively. ReLU,
being comparatively newer, is known to give better
results than other functions and Sigmoid is used in
the output layer as it is perfect for binary classification. Dropout regularization was also used to avoid
overfitting. Adam Optimizer, which almost always
outperforms other optimizers [19], was used and
binary cross entropy was used as the loss function.
This architecture was chosen as it performed well
in a similar problem [13].

## Approach 4
This approach classifies tweets using a Recurrent
neural network (LSTM)
1) Preprocessing: Special characters were
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
2) Feature extraction and Selection: The
tokenized tweets and the label column were
considered. Transfer Learning was used for
feature extraction. Glove embeddings and twitter
embeddings models were taken and used rather
than training the word vectors from scratch.
Both the embeddings were stacked together using
‘numpy.hstack’ method. The stacking demonstrated
better performance than using any one of the
embeddings or training them from scratch [8].
3) Modelling: A bi-directional LSTM neural network was used for modelling for better contextual
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
1) Preprocessing, Feature extraction, Feature
selection and Dimensionality reduction: The four
approaches stated in A,B,C and D were used as
sub models of this ensemble model. The overview
is portrayed in Fig. 5.
2) Modelling: The predictions from all 4 models
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


