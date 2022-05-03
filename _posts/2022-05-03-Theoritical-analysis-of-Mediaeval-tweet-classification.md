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
