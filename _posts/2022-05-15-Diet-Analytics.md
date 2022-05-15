---
layout: post
title: Diet and Nutrition Data Analysis
subtitle:
cover-img: /assets/img/diet_post_cover.jpg
thumbnail-img: /assets/img/diet_thumb.png
share-img: /assets/img/diet_post_cover.jpg
tags: [Data Science, Data Analysis, Data Visualization]
---

I am passionate about fitness. Nutrition is a crucial part of fitness and hence, I love learning about the way nutrition works, and about people's habits and trends when it comes to food. 
I have have gone through quite a bit of fitness content available online and felt like it was time I did some analysis of my own.

# Dataset used
The dataset used for this was originally obtained from MyFitnessPal. It contains 587,187 days of food diary records logged by 9.9K users over a 208 day period from September 2014 through April 2015. The time-series data includes a unique anonymised user id, date, list of foods and their corresponding macro-nutrient breakdown. Hats off to [Ingmar Weber and Palakorn Achananuparp](https://ink.library.smu.edu.sg/sis_research/4380/) for making this data available.

# Exploratory data analysis
## Food demand analysis
First, I wanted to find the top foods consumed overall. I developed a function which picked them out month-wise and the results were as follows:

![image](https://user-images.githubusercontent.com/26760537/168484800-1e548a88-5aca-4aaa-8062-8f177dbbf213.png)

It was obvious from this chart, how much people love their favourite cuppa. Coffee was the unparallelled choice through the year. The simple reason behind this could be that coffee is not just a drink. It is one of those simple pleasures that people enjoy as a part of their daily routine. It is also something that brings people together. 
Following this were bananas. Bananas were consumed widely on a daily basis and were the most consumed fruits. Strawberries were also widely consumed but were superseded by bananas. However, this was reversed in the month of April. The reason for this might me the onset of summer.

Spinach was the most consumed vegetable and Salted butter was the most consumed fat source. This might be surprising as oil is known to be used a lot more than butter. The reason behind this result must be that there is a lot of variety when it comes to oil. Some of the popular ones are olive oil, vegetable oil and sunflower oil. This is not the case when it comes to butter and hence there is a higher chance of more people using the same butter, which could have caused this statistical anomaly.

Chicken breast was the most popular source. This comes as no surprise as it is lean, high in protein and widely available.

This analysis can aid grocery stores in effectively stocking up on groceries. Anticipating the correct amount of need can help them avoid low profits due to products going out of stock and food wastage caused by products expiring.

## User analysis
The number of active users was analysed on each day between September 2014 and April 2015. 

![image](https://user-images.githubusercontent.com/26760537/168489021-b6e0b8f8-5a40-49d3-a24b-b76e9e92c4f0.png)

A sharp drop can be seen right after day 100. This was determined to be christmas. This makes sense as this is the start of the holidays and most people spend this time on vacation or with families, and consequently tend to get off their diet. The calories and macros are not available at most family dinners to input into the app either, making tracking much harder.


Following this, there is a dramatic rise in the number of users around the new year time. The reason for this could be that a lot of people would have completely let go and put on holiday weight and hence, would be motivated to shed off the pounds after the holidays. The fact that it was a new year also motivates people to make healthy changes. Hence a large numbers of users must have started using the app seriously during this time.

There was a serious drop in users again towards the end of March. The most plausible explaination of this is that a lot of users who decided to start using MyFitnessPal on January 1, were on a 3 month diet (one of the most popular diet durations as it is not too long to induce starting anxiety and not too short to be ineffective). Also, a few of the users must have stopped due to not noticing any benefits or lack of motivation.

The weekday on which a user started their calorie tracking journey was analysed. 

![image](https://user-images.githubusercontent.com/26760537/168489880-1cb252c8-49b6-488f-ac0d-39102365b0b2.png)

Here, there was a significant difference between Monday and all the other days. This is beacause it makes sense to just start a diet on Monday. This makes for a stronger start and easier tracking. In contrast, Wednesday was much lower than all the other days. This could have been because it was the middle of the week and motivations were at an all time low. So most people would just decide to let this week pass and start from monday.

The number of days a user used the app was then visualized.

![image](https://user-images.githubusercontent.com/26760537/168489371-66354141-340d-4ca4-922f-1ff483951870.png)

A large number of users used the app for just 72 hours, with the majority of them (422) just using the app on the first day. This might be the result of users simply wanting to try the app and see how tracking calories worked, and then deciding against doing it or planning to come back to it another day.

Following this drastic drop, the users logging food gradually declines. The reason for this is most likely lack of motivation and will power. It could also be the case that the users did not notice any benefit and hence decided to quit.

However, there is a significant rise in the total number of days logged as we approach the 180 mark. 178 days was the highest with 1190 users. This could be the result of availability of a large amount of 180 days diet challenges online.

The mean of this 'days used' analysis was 59 and the median was 42. Since the mean is higher than the median, it can be inferred that the data is skewed by some users who used the app for a high number of days. It is delightful to see that there were users who were so consistent and determined that they ended up skewing the distribution.

To further analyse the consistency of users, the amount of users who missed less than 10 days, was plotted.

![image](https://user-images.githubusercontent.com/26760537/168489522-71820e24-1878-44dd-8078-167b85c7b65c.png)

It appeared that over a quarter of the users missed less than 10 days. However, this was later found to be skewed by the amount of users who used the app for less than 10 days. To further investigate this, users with less than 10 days of usage were dropped.

![image](https://user-images.githubusercontent.com/26760537/168489597-81ed6875-cd6f-4b04-864d-651164765ccd.png)

On doing this, the number fell by over a 1000 users. However, it was still quite high when compared to the rest of the distribution. Despite this, there was still strong influenced by users with very few days logged. 

Hence, I just picked out the really serious users with over 90 days logged and less than 10 days missed.

![image](https://user-images.githubusercontent.com/26760537/168489705-1db024ca-5c96-48c5-803e-46627ce238e9.png)

From this it was inferred that there were aroud 1100 people who consistently tracked food for 3 months or more and missed under 20 days during this time. These users can be considered serious and committed and make up about 1/9th of the total population in this dataset.

The number of meals was also analysed and it was noticed the most people ate 4 meals a day. This is shown below.

![image](https://user-images.githubusercontent.com/26760537/168489285-d3412317-450d-47f2-b24e-9e98a7975ca7.png)

I wanted to then find out how many users actually met their goal and how many were in a calorie defecit and on track to lose weight.

![image](https://user-images.githubusercontent.com/26760537/168489821-c3eac14e-c17d-49c7-88dd-fb58e76c0ba5.png)

It was delightful to see that just 15% of the users have significantly exceeded their calorie goal for the days logged by over 100 calories. The remaining 85% succesfully stuck to their goal overall.

Finally, the sugar intake of users was analysed.

![image](https://user-images.githubusercontent.com/26760537/168490126-eee7ae0e-9f1d-4df4-8cbf-545deb5516da.png)

The left is the lowest risk, middle being medium risk and right beign the lowest risk. It was unfortunate to find that half the userbase was at a risk of developing type II diabetes. 20% of the users were at high risk and consumed exorbident amounts of sugar, which was over 200 grams per day. 
In my opinion, sugar is one of the worst things you can put in your body. It has no nutritional benifit. Adding to this, it is addictive and an inflammatory substance. Using natural sweetners such as Stevia would greatly help in combating type 2 diabetes and help make the world a happier and healthier place. 

# Key Insights derived
Recapitulating my findings below,

Insights to aid supermarket in efficient and effective stocking of goods:
* Coffee was the most popular item consumed.
* Banana was the most consumed fruit
* The most popular protein source was chicken breast

Insights about people and nutrition:
* There was a sharp drop during christmas and a dramatic rise on January 1 (New year).
* On average, 82% of the users met their goal and 85% were in a calorie deficit.
* The percentage of users that could be considered serious users (logged over 3 months and missed less than 20 days), was 11.11%.
* 20.6% were suspected to be at risk of getting type 2 diabetes.
* Monday was the day when most people started using MyFitnessPal and Wednesday was the least.

