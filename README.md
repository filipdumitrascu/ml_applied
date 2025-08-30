# Project 2 Artificial Intelligence - Applied ML

#### Dumitrascu Filip-Teodor 333CA

## Contents
1. [Introduction](#introduction)  
2. [EDA](#eda)  
3. [Data Preprocessing](#data-preprocessing)  
4. [Algorithms](#algorithms)  
5. [Comparison](#comparison) 

## Introduction
In the current activity of an engineer or researcher in the field of artificial intelligence, especially machine learning, the following three essential components frequently appear:

- Visualization and exploratory data analysis (EDA – Exploratory Data Analysis)  
- Identifying and extracting relevant data features, useful for the proposed objective (such as classification, regression, or anomaly detection)  
- Testing and comparing multiple models, in order to choose the most efficient solution for the addressed problem.  

Thus, in this project two datasets are analyzed (Air Pollution and News Popularity), which are processed through the steps mentioned above, aiming to deepen the understanding of the machine learning process.

## EDA
### Air pollution
#### Attribute types
Considering the type of values they contain and the ranges in which they vary, the dataset attributes can be classified into the following categories (ignoring `Unnamed 0`):  
- continuous attributes: `AQI_Value`, `CO_Value`, `Ozone_Value`, `NO2_Value`, `PM25_Value`, `VOCs`, `SO2` (numeric, unique values > 20, value range > 50)  
- discrete attributes: `Country`, `City` (the rest that are neither continuous nor ordinal)  
- ordinal attributes: `AQI_Category`, `CO_Category`, `Ozone_Category`, `NO2_Category`, `PM25_Category`, `Emissions` (contain ordered objects: Good–Moderate, Level0–Level5)  

#### Continuous numerical attributes analysis
| Attribute   | Count   | Mean      | Std Dev   | Min     | 25%     | 50%     | 75%     | Max       |
|-------------|---------|-----------|-----------|---------|---------|---------|---------|-----------|
| AQI_Value   | 23463.0 | 72.01     | 56.06     | 6.00    | 39.00   | 55.00   | 79.00   | 500.00    |
| CO_Value    | 23463.0 | 1.37      | 1.83      | 0.00    | 1.00    | 1.00    | 1.00    | 133.00    |
| Ozone_Value | 21117.0 | 35.24     | 28.15     | 0.00    | 21.00   | 31.00   | 40.00   | 222.00    |
| NO2_Value   | 23463.0 | 43.08     | 196.08    | 0.00    | 0.00    | 1.00    | 4.00    | 1003.06   |
| PM25_Value  | 23463.0 | 68.52     | 54.80     | 0.00    | 35.00   | 54.00   | 79.00   | 500.00    |
| VOCs        | 23463.0 | 185.05    | 140.49    | 12.42   | 103.27  | 142.97  | 204.23  | 1280.99   |
| SO2         | 23463.0 | 4.45      | 5.95      | -18.53  | 0.74    | 4.29    | 7.92    | 234.69    |

![alt text](images/air_polution/boxplot/boxplot_cont_feat_air.png)

The boxplot highlights that most values for continuous attributes in the Air Pollution dataset (such as `SO2`, `CO_Value`, `NO2_Value`) are concentrated in relatively small ranges, while there are numerous outliers that far exceed these values. The attributes `VOCs`, `PM25_Value`, and `AQI_Value` are notable for their wide ranges of values and high number of extreme values, indicating an unbalanced distribution and a possible need for outlier treatment.

#### Ordinal attributes analysis

- AQI_Category — Number of examples non-null: 23463
- AQI_Category — Number of unique values: 6
![alt text](images/air_polution/histogram/ordinal_feat_histogram_AQI_Category.png)
AQI_Category shows a highly skewed distribution, with most items labeled as Good or Moderate. The Very Unhealthy and Hazardous classes are significantly underrepresented.

- CO_Category — Number of examples non-null: 21117
- CO_Category — Number of unique values: 2
![alt text](images/air_polution/histogram/ordinal_feat_histogram_CO_Category.png)

- Ozone_Category — Number of examples non-null: 23463
- Ozone_Category — Number of unique values: 5
![alt text](images/air_polution/histogram/ordinal_feat_histogram_Emissions.png)

- NO2_Category — Number of examples non-null: 23463
- NO2_Category — Number of unique values: 2
![alt text](images/air_polution/histogram/ordinal_feat_histogram_NO2_Category.png)
CO_Category, NO2_Category, and Ozone_Category are dominated by the Good category, with the other classes being almost absent. This imbalance may affect the performance of classification algorithms.

- PM25_Category — Number of examples non-null: 23463
- PM25_Category — Number of unique values: 6
![alt text](images/air_polution/histogram/ordinal_feat_histogram_Ozone_Category.png)

- Emissions — Number of examples non-null: 23463
- Emissions — Number of unique values: 6
![alt text](images/air_polution/histogram/ordinal_feat_histogram_PM25_Category.png)
Emissions and PM25_Category show a more varied distribution, but still with a high concentration in classes L0 and L1, indicating a moderate to severe imbalance within the L-type ordinal classes.

#### Class Balance
![alt text](images/air_polution/boxplot/boxplot_classes_good-hazardous_air.png)

The graph highlights a significant imbalance between the AQI categories. Most examples are labeled as "Good" (over 70,000), followed at a considerable distance by "Moderate" and then by the other classes ("Unhealthy for Sensitive Groups," "Unhealthy," "Very Unhealthy," and "Hazardous"), which are underrepresented. This imbalance can affect the performance of classification algorithms, which will lean toward the dominant classes.

![alt text](images/air_polution/boxplot/boxplot_classes_l0-l5_air.png)

The ordinal attributes related to emission levels (L0–L5) also show an imbalance: classes L0 and L1 are the most frequent, while L4 and L5 occur rarely. This type of distribution can negatively influence the model's ability to learn to correctly recognize less frequent cases (high emission levels).

#### Corelatia intre atribute

![alt text](images/air_polution/correlation/correlation_cont_matrix.png)

This matrix highlights the linear correlations between the continuous numerical attributes in the Air dataset. There is an almost perfect correlation (≈1.00) between AQI_Value, PM25_Value, and VOCs, indicating informational redundancy. In contrast, NO2_Value has correlations close to 0 with all other attributes, suggesting that it is linearly independent from the rest.

![alt text](images/air_polution/correlation/correlation_ordinal_matrix.png)

The matrix shows the p-values obtained from the Chi-square test applied to pairs of categorical variables. Values close to 0 indicate a statistically significant association (correlation), while high values (e.g., CO_Category and NO2_Category) suggest a lack of association. For example, AQI_Category has significant links to all other attributes, reflecting its central importance in air quality assessment.

### News pollution
#### Attribute type
Given the type of values they contain and the ranges in which they vary, the attributes in the dataset can be classified into the following categories: (ignore `url`)
- continuous attributes: ` days_since_published`, ` content_word_count`, ` unique_word_ratio`, ` non_stop_word_ratio`, ` unique_non_stop_ratio`, `external_links`, `internal_links`, `image_count`, `video_count`, `keyword_worst_min_shares`, `keyword_worst_max_shares`, `keyword_worst_avg_shares`, `keyword_best_min_shares`, `keyword_best_max_shares`, `keyword_best_avg_shares`, `keyword_avg_min_shares`, ` keyword_avg_max_shares`, ` keyword_avg_avg_shares`, ` ref_min_shares`, ` ref_max_shares`, ` ref_avg_shares`, ` engagement_ratio`, ` content_density` (numeric, unique values > 20, value range > 50)
- discrete attributes: ` title_word_count`, ` avg_word_length`, ` keyword_count`, ` topic_0_relevance`, ` topic_1_relevance`, ` topic_2_relevance`, ` topic_3_relevance`, ` topic_4_relevance`, ` content_subjectivity`, ` content_sentiment`, ` positive_word_rate`, ` negative_word_rate`, ` non_neutral_positive_rate`, `non_neutral_negative_rate`, `avg_positive_sentiment`, `min_positive_sentiment`, `max_positive_sentiment`, `avg_negative_sentiment`, `min_negative_sentiment`, `max_negative_sentiment`, `title_subjectivity`, `title_sentiment`, `title_subjectivity_magnitude`, `title_sentiment_magnitude` (numeric, rest)
- ordinal attributes: `popularity_category` (contains ordered objects: Slightly Popular, Moderately Popular)
- categorical attributes: `channel_lifestyle`, `channel_entertainment`, `channel_business`, `channel_social_media`, `channel_tech`, `channel_world`, `day_monday`, `day_tuesday`, `day_wednesday`, `day_thursday`, `day_friday`, `day_saturday`, `day_sunday`, `is_weekend`, `publication_period` (rest of objects)

#### Continuous numerical attributes analysis
| Attribute                  | Count     | Mean       | Std Dev   | Min       | 25%       | 50%       | 75%       | Max         |
|----------------------------|-----------|------------|-----------|-----------|-----------|-----------|-----------|-------------|
| days_since_published       | 39644.0   | 354.53     | 214.16    | 8.00      | 164.00    | 339.00    | 542.00    | 731.00      |
| content_word_count         | 39644.0   | 546.51     | 471.11    | 0.00      | 246.00    | 409.00    | 716.00    | 8474.00     |
| unique_word_ratio          | 39644.0   | 0.55       | 3.52      | 0.00      | 0.47      | 0.54      | 0.61      | 701.00      |
| non_stop_word_ratio        | 39644.0   | 1.00       | 5.23      | 0.00      | 1.00      | 1.00      | 1.00      | 1042.00     |
| unique_non_stop_ratio      | 39644.0   | 0.69       | 3.26      | 0.00      | 0.63      | 0.69      | 0.75      | 650.00      |
| external_links             | 39644.0   | 192.25     | 905.42    | 0.00      | 4.00      | 8.00      | 15.00     | 6078.62     |
| internal_links             | 39644.0   | 3.29       | 3.86      | 0.00      | 1.00      | 3.00      | 4.00      | 116.00      |
| image_count                | 39644.0   | 4.54       | 8.31      | 0.00      | 1.00      | 1.00      | 4.00      | 128.00      |
| video_count                | 39644.0   | 1.25       | 4.11      | 0.00      | 0.00      | 0.00      | 1.00      | 91.00       |
| keyword_worst_min_shares   | 39644.0   | 26.11      | 69.63     | -1.00     | -1.00     | -1.00     | 4.00      | 377.00      |
| keyword_worst_max_shares   | 39644.0   | 1153.95    | 3857.99   | 0.00      | 445.00    | 660.00    | 1000.00   | 298400.00   |
| keyword_worst_avg_shares   | 39644.0   | 312.37     | 620.78    | -1.00     | 141.75    | 235.50    | 357.00    | 42827.86    |
| keyword_best_min_shares    | 39644.0   | 13612.35   | 57986.03  | 0.00      | 0.00      | 1400.00   | 7900.00   | 843300.00   |
| keyword_best_max_shares    | 39644.0   | 752324.07  | 214502.13 | 0.00      | 843300.00 | 843300.00 | 843300.00 | 843300.00   |
| keyword_best_avg_shares    | 39644.0   | 259281.94  | 135102.25 | 0.00      | 172846.88 | 244572.22 | 330980.00 | 843300.00   |
| keyword_avg_min_shares     | 39644.0   | 1117.15    | 1137.46   | -1.00     | 0.00      | 1023.64   | 2056.78   | 3613.04     |
| keyword_avg_max_shares     | 39644.0   | 5657.21    | 6098.87   | 0.00      | 3562.10   | 4355.69   | 6019.95   | 298400.00   |
| keyword_avg_avg_shares     | 39644.0   | 3135.86    | 1318.15   | 0.00      | 2382.45   | 2870.07   | 3600.23   | 43567.66    |
| ref_min_shares             | 39644.0   | 3998.76    | 19738.67  | 0.00      | 639.00    | 1200.00   | 2600.00   | 843300.00   |
| ref_max_shares             | 39644.0   | 10329.21   | 41027.58  | 0.00      | 1100.00   | 2800.00   | 8000.00   | 843300.00   |
| ref_avg_shares             | 39644.0   | 6401.70    | 24211.33  | 0.00      | 981.19    | 2200.00   | 5200.00   | 843300.00   |
| engagement_ratio           | 39644.0   | 1054.07    | 3496.61   | 0.04      | 220.00    | 465.50    | 900.00    | 221200.00   |
| content_density            | 35680.0   | 1986.56    | 2209.10   | 32.76     | 746.07    | 1314.55   | 2506.71   | 58857.97    |


![alt text](images/news_popularity/boxplot/boxplot_cont_feat_news.png)

The boxplot for the News Popularity set shows high variability between attributes, with some of them having extremely high values (over 800,000), such as `keyword_best_max_shares` or `ref_max_shares`. Many of the attributes are highly unbalanced, with low medians and the presence of extreme outliers. This suggests that a stage of treating extreme values and possibly standardizing the data is necessary to avoid influencing the ML algorithms.

#### Ordinal/categorical attributes analysis

- channel_business — Number of non-null examples: 39644
- channel_business — Number of unique values: 2
![alt text](<images/news_popularity/histogram/histogram_categ_feat_news_ channel_business.png>)


- channel_entertainment — Number of non-null examples: 39644
- channel_entertainment — Number of unique values: 2
![alt text](<images/news_popularity/histogram/histogram_categ_feat_news_ channel_entertainment.png>)


- channel_lifestyle — Number of non-null examples: 35680
- channel_lifestyle — Number of unique values: 2
![alt text](<images/news_popularity/histogram/histogram_categ_feat_news_ channel_lifestyle.png>)


- channel_social_media — Number of non-null examples: 39644
- channel_social_media — Number of unique values: 2
![alt text](<images/news_popularity/histogram/histogram_categ_feat_news_ channel_social_media.png>)


- channel_tech — Number of non-null examples: 39644
- channel_tech — Number of unique values: 2
![alt text](<images/news_popularity/histogram/histogram_categ_feat_news_ channel_tech.png>)

- channel_world — Number of non-null examples: 39644
- channel_world — Number of unique values: 2
![alt text](<images/news_popularity/histogram/histogram_categ_feat_news_ channel_world.png>)

Binary attributes such as channel_business, channel_entertainment, channel_lifestyle, channel_social_media, channel_tech, channel_world indicate whether an article belongs to a particular thematic channel. The distribution is unbalanced in all cases: 'N' values (does not belong to the channel) predominate, and only a relatively small fraction of articles are marked with 'Y' (yes). This can influence the performance of classifiers, as the model learns more slowly from rare classes.

- day_monday — Number of non-null examples: 39644
- day_monday — Number of unique values: 2
![alt text](<images/news_popularity/histogram/histogram_categ_feat_news_ day_monday.png>)


- day_tuesday — Number of non-null examples: 39644
- day_tuesday — Number of unique values: 2
![alt text](<images/news_popularity/histogram/histogram_categ_feat_news_ day_tuesday.png>)


- day_wednesday — Number of non-null examples: 39644
- day_wednesday — Number of unique values: 2
![alt text](<images/news_popularity/histogram/histogram_categ_feat_news_ day_wednesday.png>)

- day_thursday — Number of non-null examples: 39644
- day_thursday — Number of unique values: 2
![alt text](<images/news_popularity/histogram/histogram_categ_feat_news_ day_thursday.png>)

- day_friday — Number of non-null examples: 39644
- day_friday — Number of unique values: 2
![alt text](<images/news_popularity/histogram/histogram_categ_feat_news_ day_friday.png>)

- day_saturday — Number of non-null examples: 39644
- day_saturday — Number of unique values: 2
![alt text](<images/news_popularity/histogram/histogram_categ_feat_news_ day_saturday.png>)

- day_sunday — Number of non-null examples: 39644
- day_sunday — Number of unique values: 2
![alt text](<images/news_popularity/histogram/histogram_categ_feat_news_ day_sunday.png>)

- is_weekend — Number of non-null examples: 39644
- is_weekend — Number of unique values: 2
![alt text](<images/news_popularity/histogram/histogram_categ_feat_news_ is_weekend.png>)


Binary attributes such as day_monday, day_friday, day_saturday, etc. indicate whether the article was published on a specific day. The distribution is also unbalanced—most values are 'N', while 'Y' only appears when the article was published on that day. There is a slight variation between weekdays and weekends.

- publication_period — Number of non-null examples: 39644
- publication_period — Number of unique values: 2
![alt text](<images/news_popularity/histogram/histogram_categ_feat_news_ publication_period.png>)

Most articles are published during the week (Weekday), with a smaller proportion on the Weekend.

- popularity_category — Number of non-null examples: 39644
- popularity_category — Number of unique values: 5
![alt text](images/news_popularity/histogram/histogram_categ_feat_news_popularity_category.png)

The most common class is Slightly Popular, followed by Moderately Popular. Extreme classes, such as Very Unpopular, are rare, suggesting unbalanced labeling and potential difficulty in classification.

#### Class Balance
![alt text](images/news_popularity/boxplot/boxplot_classes_unpopular-popular.png)

The graph shows a significant imbalance between classes: most articles are labeled as Slightly Popular and Moderately Popular, while the extreme classes (Very Unpopular, Popular) are underrepresented. This imbalance can affect classification algorithms, which tend to favor dominant classes.

![alt text](images/news_popularity/boxplot/boxplot_classes_weekday-weekend.png)

Most articles are published during the week (Weekday), which reflects common practice in online media. Weekend articles represent a much smaller percentage, suggesting a potential seasonal effect or different audience behavior.

![alt text](images/news_popularity/boxplot/boxplot_classes_yes-no_combined.png)

The total distribution of Y and N values for all binary columns (such as channel_*, day_*) is deeply skewed towards the value N. This indicates that articles rarely belong to a specific channel or day. This imbalance must be carefully addressed in classification models (e.g., through weighting or sampling).

#### Correlation between attributes

![alt text](images/news_popularity/correlation/correlation_cont_matrix_news.png)
This matrix shows Pearson correlations between continuous numerical attributes. Very strong correlations can be observed between:
`content_word_count` and `content_density` (0.97) – longer articles tend to have higher content density.
The variables related to keyword shares (keyword_*) are positively correlated with each other (values between 0.4 and 0.8).
`ref_avg_shares` and `ref_max_shares` or `ref_min_shares` are also moderately to strongly correlated.
In general, there are clear groups of correlated variables, useful for dimensionality reduction or attribute selection.


![alt text](images/news_popularity/correlation/correlation_ordinal_matrix.png)
This matrix expresses the statistical significance of the relationship between categorical attributes (including ordinal ones) using the Chi-square test (p-value).
Values close to 0 (bright yellow) indicate a statistically significant relationship between variables.
Strong correlations can be observed between:
Days of the week (`day_monday`, `day_tuesday`, etc.) and `is_weekend` or `publication_period`.
Channels (`channel_*`) and `publication_period`, suggesting that some types of content are more common in certain periods.
`popularity_category` is significantly associated with most other categorical variables, suggesting that the popularity of articles is influenced by many discrete factors.


## Data Preprocessing
### Air pollution data set

#### Attributes with missing data and their imputation
Atributes with missing values:
CO_Category    1893
Ozone_Value    1870
Country         349

After imputation, missing values in train:
CO_Category    0
Ozone_Value    0
Country        0

#### Extreme values and their replacement with those in the allocation
[Train] AQI_Value: 2354 outliers replaced with NaN
[Train] CO_Value: 6858 outliers replaced with NaN
[Train] Ozone_Value: 1219 outliers replaced with NaN
[Train] NO2_Value: 2038 outliers replaced with NaN
[Train] PM25_Value: 2160 outliers replaced with NaN
[Train] VOCs: 2331 outliers replaced with NaN
[Train] SO2: 240 outliers replaced with NaN

Replacing outliers with NaN in airDataTest using train IQR...
[Test] AQI_Value: 581 outliers replaced with NaN
[Test] CO_Value: 1738 outliers replaced with NaN
[Test] Ozone_Value: 284 outliers replaced with NaN
[Test] NO2_Value: 516 outliers replaced with NaN
[Test] PM25_Value: 547 outliers replaced with NaN
[Test] VOCs: 576 outliers replaced with NaN
[Test] SO2: 56 outliers replaced with NaN

Outliers replaced with NaN and imputed with column mean.

#### Redundant variables
Removed redundant features: ['VOCs', 'PM25_Value', 'CO_Category', 'Ozone_Category', 'NO2_Category', 'PM25_Category', 'Emissions', 'City', 'Country']

#### Standardization
For Logistic Regression to work, the values of numerical attributes are standardized.

### News pollution data set

#### Attributes with missing data and their imputation
Atributes with missing values:
channel_lifestyle    3175
content_density      3145

After imputation, missing values in train:
channel_lifestyle    0
content_density      0


#### Extreme values and their replacement with those in the allocation
[Train]  days_since_published: 0 outliers replaced with NaN
[Train]  title_word_count: 123 outliers replaced with NaN
[Train]  content_word_count: 1539 outliers replaced with NaN
[Train]  unique_word_ratio: 1290 outliers replaced with NaN
[Train]  non_stop_word_ratio: 2234 outliers replaced with NaN
[Train]  unique_non_stop_ratio: 1393 outliers replaced with NaN
[Train]  external_links: 2665 outliers replaced with NaN
[Train]  internal_links: 1674 outliers replaced with NaN
[Train]  image_count: 6159 outliers replaced with NaN
[Train]  video_count: 2330 outliers replaced with NaN
[Train]  avg_word_length: 1363 outliers replaced with NaN
[Train]  keyword_count: 47 outliers replaced with NaN
[Train]  keyword_worst_min_shares: 3743 outliers replaced with NaN
[Train]  keyword_worst_max_shares: 2939 outliers replaced with NaN
[Train]  keyword_worst_avg_shares: 1694 outliers replaced with NaN
[Train]  keyword_best_min_shares: 4025 outliers replaced with NaN
[Train]  keyword_best_max_shares: 7542 outliers replaced with NaN
[Train]  keyword_best_avg_shares: 791 outliers replaced with NaN
[Train]  keyword_avg_min_shares: 0 outliers replaced with NaN
[Train]  keyword_avg_max_shares: 1893 outliers replaced with NaN
[Train]  keyword_avg_avg_shares: 1284 outliers replaced with NaN
[Train]  ref_min_shares: 3979 outliers replaced with NaN
[Train]  ref_max_shares: 3450 outliers replaced with NaN
[Train]  ref_avg_shares: 3362 outliers replaced with NaN
[Train]  topic_0_relevance: 4238 outliers replaced with NaN
[Train]  topic_1_relevance: 4624 outliers replaced with NaN
[Train]  topic_2_relevance: 2864 outliers replaced with NaN
[Train]  topic_3_relevance: 211 outliers replaced with NaN
[Train]  topic_4_relevance: 0 outliers replaced with NaN
[Train]  content_subjectivity: 1551 outliers replaced with NaN
[Train]  content_sentiment: 682 outliers replaced with NaN
[Train]  positive_word_rate: 415 outliers replaced with NaN
[Train]  negative_word_rate: 1052 outliers replaced with NaN
[Train]  non_neutral_positive_rate: 1289 outliers replaced with NaN
[Train]  non_neutral_negative_rate: 390 outliers replaced with NaN
[Train]  avg_positive_sentiment: 1701 outliers replaced with NaN
[Train]  min_positive_sentiment: 2527 outliers replaced with NaN
[Train]  max_positive_sentiment: 0 outliers replaced with NaN
[Train]  avg_negative_sentiment: 704 outliers replaced with NaN
[Train]  min_negative_sentiment: 0 outliers replaced with NaN
[Train]  max_negative_sentiment: 1987 outliers replaced with NaN
[Train]  title_subjectivity: 0 outliers replaced with NaN
[Train]  title_sentiment: 6319 outliers replaced with NaN
[Train]  title_subjectivity_magnitude: 0 outliers replaced with NaN
[Train]  title_sentiment_magnitude: 1343 outliers replaced with NaN
[Train]  engagement_ratio: 3225 outliers replaced with NaN
[Train]  content_density: 2355 outliers replaced with NaN

Replacing outliers with NaN in newsDataTrain using train IQR...
[Test]  days_since_published: 0 outliers replaced with NaN
[Test]  title_word_count: 0 outliers replaced with NaN
[Test]  content_word_count: 0 outliers replaced with NaN
[Test]  unique_word_ratio: 0 outliers replaced with NaN
[Test]  non_stop_word_ratio: 0 outliers replaced with NaN
[Test]  unique_non_stop_ratio: 0 outliers replaced with NaN
[Test]  external_links: 0 outliers replaced with NaN
[Test]  internal_links: 0 outliers replaced with NaN
[Test]  image_count: 0 outliers replaced with NaN
[Test]  video_count: 0 outliers replaced with NaN
[Test]  avg_word_length: 0 outliers replaced with NaN
[Test]  keyword_count: 0 outliers replaced with NaN
[Test]  keyword_worst_min_shares: 0 outliers replaced with NaN
[Test]  keyword_worst_max_shares: 0 outliers replaced with NaN
[Test]  keyword_worst_avg_shares: 0 outliers replaced with NaN
[Test]  keyword_best_min_shares: 0 outliers replaced with NaN
[Test]  keyword_best_max_shares: 0 outliers replaced with NaN
[Test]  keyword_best_avg_shares: 0 outliers replaced with NaN
[Test]  keyword_avg_min_shares: 0 outliers replaced with NaN
[Test]  keyword_avg_max_shares: 0 outliers replaced with NaN
[Test]  keyword_avg_avg_shares: 0 outliers replaced with NaN
[Test]  ref_min_shares: 0 outliers replaced with NaN
[Test]  ref_max_shares: 0 outliers replaced with NaN
[Test]  ref_avg_shares: 0 outliers replaced with NaN
[Test]  topic_0_relevance: 0 outliers replaced with NaN
[Test]  topic_1_relevance: 0 outliers replaced with NaN
[Test]  topic_2_relevance: 0 outliers replaced with NaN
[Test]  topic_3_relevance: 0 outliers replaced with NaN
[Test]  topic_4_relevance: 0 outliers replaced with NaN
[Test]  content_subjectivity: 0 outliers replaced with NaN
[Test]  content_sentiment: 0 outliers replaced with NaN
[Test]  positive_word_rate: 0 outliers replaced with NaN
[Test]  negative_word_rate: 0 outliers replaced with NaN
[Test]  non_neutral_positive_rate: 0 outliers replaced with NaN
[Test]  non_neutral_negative_rate: 0 outliers replaced with NaN
[Test]  avg_positive_sentiment: 0 outliers replaced with NaN
[Test]  min_positive_sentiment: 0 outliers replaced with NaN
[Test]  max_positive_sentiment: 0 outliers replaced with NaN
[Test]  avg_negative_sentiment: 0 outliers replaced with NaN
[Test]  min_negative_sentiment: 0 outliers replaced with NaN
[Test]  max_negative_sentiment: 0 outliers replaced with NaN
[Test]  title_subjectivity: 0 outliers replaced with NaN
[Test]  title_sentiment: 0 outliers replaced with NaN
[Test]  title_subjectivity_magnitude: 0 outliers replaced with NaN
[Test]  title_sentiment_magnitude: 0 outliers replaced with NaN
[Test]  engagement_ratio: 0 outliers replaced with NaN
[Test]  content_density: 0 outliers replaced with NaN

Outliers replaced with NaN and imputed with column mean.

#### Redundant variables
Removed redundant features: ['content_density', 'non_stop_word_ratio', 'unique_non_stop_ratio', 'keyword_best_avg_shares', 'keyword_avg_avg_shares', 'ref_avg_shares', 'publication_period', 'is_weekend', 'channel_business', 'channel_entertainment', 'channel_lifestyle', 'channel_social_media', 'channel_tech', 'channel_world', 'day_monday', 'day_tuesday', 'day_wednesday', 'day_thursday', 'day_friday', 'day_saturday', 'day_sunday']

#### Standardization
For Logistic Regression to work, the values of numerical attributes are standardized.

## Algorithms
### Air pollution data set
#### Decision trees

![alt text](images/air_polution/algorithms/decision_tree_air.png)

Classification Report — Decision Tree
| Class                             | Precision | Recall | F1-score | Support |
|----------------------------------|-----------|--------|----------|---------|
| Good                             | 1.00      | 1.00   | 1.00     | 1987    |
| Hazardous                        | 0.13      | 0.50   | 0.20     | 38      |
| Moderate                         | 1.00      | 1.00   | 1.00     | 1846    |
| Unhealthy                        | 0.00      | 0.00   | 0.00     | 446     |
| Unhealthy for Sensitive Groups   | 1.00      | 0.88   | 0.93     | 318     |
| Very Unhealthy                   | 0.09      | 0.67   | 0.16     | 58      |
|                                  |           |        |          |         |
| **Accuracy**                     |           |        | **0.89** | 4693    |
| **Macro Avg**                    | 0.54      | 0.67   | 0.55     | 4693    |
| **Weighted Avg**                 | 0.89      | 0.89   | 0.88     | 4693    |
Accuracy (Decision Tree): 0.8886

A decision tree model using the scikit-learn library, with the aim of predicting air quality category (AQI_Category).
The model was trained on the processed dataset (X_train, y_train), with the following hyperparameters:
- Criterion: entropy — to maximize the information gained at each split
- Maximum depth (max_depth): 4 — to avoid overfitting
- Minimum number of examples in a leaf (min_samples_leaf): 5 — to avoid sparse leaves
- Class weighting (class_weight): balanced — to counteract the unbalanced distribution of classes

Although the model achieves high overall accuracy (88.86%), the precision and recall scores for underrepresented classes (e.g., "Hazardous," "Unhealthy") are very low. This indicates that the model tends to favor frequent classes and is inefficient at correctly identifying rare ones. In a real-world context, this can be problematic if these classes represent critical conditions.

#### Random Forests

![alt text](images/air_polution/algorithms/decision_tree_air.png)

| Class                            | Precision | Recall | F1-score | Support |
|----------------------------------|-----------|--------|----------|---------|
| Good                             | 1.00      | 1.00   | 1.00     | 1987    |
| Hazardous                        | 0.08      | 0.71   | 0.14     | 38      |
| Moderate                         | 0.98      | 0.94   | 0.96     | 1846    |
| Unhealthy                        | 0.87      | 0.26   | 0.40     | 446     |
| Unhealthy for Sensitive Groups   | 1.00      | 0.88   | 0.93     | 318     |
| Very Unhealthy                   | 0.12      | 0.36   | 0.19     | 58      |
|                                  |           |        |          |         |
| **Accuracy**                     |           |        | **0.89** | 4693    |
| **Macro Avg**                    | 0.67      | 0.69   | 0.60     | 4693    |
| **Weighted Avg**                 | 0.96      | 0.89   | 0.91     | 4693    |
Accuracy (Random Forest): 0.8881

A Random Forest model was trained using the scikit-learn library to predict air quality category (AQI_Category). The model was trained on the processed data (X_train, y_train) and used the following hyperparameters:

- Criterion: entropy — maximizes information gain at each split;
- Maximum depth (max_depth): 4 — limits the complexity of each tree to prevent overfitting;
- Minimum number of examples in a leaf (min_samples_leaf): 5 — prevents the formation of leaves with few instances;
- Class weighting (class_weight): balanced — automatically adjusts weights according to class distribution, which is useful for unbalanced sets;
- Number of estimators (n_estimators): 100 — uses 100 individual decision trees;
- Sample proportion for each estimator: default (bootstrap=True) — each tree is trained on a different bootstrap subset of the data;
- Proportion of attributes used by each tree: default (max_features='sqrt') — each tree randomly selects √n attributes at each split, which helps reduce correlation between trees and improve generalization.

The model achieves an overall accuracy of 88.9%, but performance on imbalanced classes remains limited. For example, for the "Hazardous" class, the model has a recall score of 71%, but the precision is only 8%, indicating a high number of false positives. Also, classes such as "Unhealthy" or "Very Unhealthy" have difficulty being classified correctly. This situation suggests that, although Random Forest slightly improves performance over the simple decision tree, challenges related to class imbalance persist. In critical applications (e.g., extreme pollution), these limitations can affect the system's ability to respond appropriately to dangerous conditions.

#### Logistic Regression

![alt text](images/air_polution/algorithms/logistic_regression_air.png)

| Class                                | Precision | Recall | F1-score | Support |
|--------------------------------------|-----------|--------|----------|---------|
| Good                                 | 0.81      | 1.00   | 0.90     | 1987    |
| Hazardous                            | 0.00      | 0.00   | 0.00     | 38      |
| Moderate                             | 0.74      | 0.57   | 0.64     | 1846    |
| Unhealthy                            | 0.50      | 0.52   | 0.51     | 446     |
| Unhealthy for Sensitive Groups       | 0.02      | 0.02   | 0.02     | 318     |
| Very Unhealthy                       | 0.00      | 0.00   | 0.00     | 58      |
|                                      |           |        |          |         |
| **Accuracy**                         |           |        | **0.70** | 4693    |
| **Macro Avg**                        | 0.34      | 0.35   |  0.34    | 4693    |
| **Weighted Avg**                     | 0.68      | 0.70   |  0.68    | 4693    |

Accuracy (Logistic Regression): 0.6953

A logistic regression model was manually implemented and trained to classify air quality (AQI_Category) using a gradient descent optimizer. This model was tested on a dataset with multiple classes, including "Good," "Moderate," "Unhealthy," and "Hazardous." For data preprocessing:
- Categorical attributes were transformed using One-Hot Encoding, applied exclusively to columns with low cardinality (via ColumnTransformer).
- Numeric attributes were kept unchanged (remainder='passthrough').
The optimization algorithm used:
- Full-batch Gradient Descent
- Learning rate: 0.1
- Number of epochs: 3000
- Weight initialization: normal distribution (N(0,1))
- Regularization: L2 (with λ = 0.01) to penalize large coefficients and reduce overfitting
The model achieved an overall accuracy of 69.5%, which is a reasonable result for a linear model applied to an unbalanced set.
However, the performance per class highlights the limitations:
- The "Good" class is classified excellently (recall 100%, f1-score 0.90)
- The "Very Unhealthy" and "Hazardous" classes are not correctly identified (accuracy and recall 0) due to the imbalance and the small number of examples
This distribution suggests that logistic regression has difficulty correctly separating rarer classes or those that overlap in the attribute space. For critical environmental applications, this limits the usefulness of the model without additional balancing methods.

#### MLP

![alt text](images/air_polution/algorithms/mlp_air.png)

| Class                             | Precision | Recall | F1-score | Support |
|----------------------------------|-----------|--------|----------|---------|
| Good                             | 1.00      | 1.00   | 1.00     | 1987    |
| Hazardous                        | 0.00      | 0.00   | 0.00     | 38      |
| Moderate                         | 0.99      | 0.98   | 0.98     | 1846    |
| Unhealthy                        | 0.72      | 0.99   | 0.83     | 446     |
| Unhealthy for Sensitive Groups   | 1.00      | 0.84   | 0.91     | 318     |
| Very Unhealthy                   | 0.00      | 0.00   | 0.00     | 58      |
|                                  |           |        |          |         |
| **Accuracy**                     |           |        | **0.96** | 4693    |
| **Macro Avg**                    | 0.62      | 0.63   | 0.62     | 4693    |
| **Weighted Avg**                 | 0.95      | 0.96   | 0.95     | 4693    |
Accuracy (MLP): 0.9593

An MLP (Multi-Layer Perceptron) neural network model was trained to predict air quality class (AQI_Category). The model was trained on the processed dataset with the following hyperparameters:
- Architecture: a single hidden layer with 100 neurons
- Activation function: ReLU

- Optimizer: Adam
- Learning rate: 0.001
- Maximum number of epochs: 200

- Early stopping criterion (early_stopping=True) to prevent overfitting
- Regularization coefficient: alpha=0.0001

The model achieved a high overall accuracy of 95.93%, but performance is heavily skewed between classes:
- Major classes such as Good, Moderate, or Unhealthy for Sensitive Groups are classified correctly in a very high proportion.
- Rare classes such as Hazardous and Very Unhealthy have 0 precision and recall, which means that the model does not identify them correctly at all.
- The Unhealthy class has very good recall (0.99) but low precision (0.72), indicating the existence of many false positives.

The model is effective in identifying frequent classes, but insufficient for critical applications where correct recognition of dangerous conditions (e.g., Hazardous) is essential.

![alt text](images/air_polution/algorithms/mlp_loss_curve.png)

![alt text](images/air_polution/algorithms/mlp_accuracy_curve.png)

### News pollution data set
#### Decision trees

![alt text](images/news_popularity/algorithms/decision_tree_news.png)

| Class                | Precision | Recall | F1-score | Support |
|---------------------|-----------|--------|----------|---------|
| Moderately Popular  | 0.87      | 0.80   | 0.83     | 2401    |
| Popular             | 0.71      | 0.71   | 0.71     | 1074    |
| Slightly Popular    | 0.87      | 0.92   | 0.90     | 3799    |
| Unpopular           | 0.33      | 0.80   | 0.47     | 218     |
| Viral               | 0.84      | 0.24   | 0.37     | 437     |
|                     |           |        |          |         |
| **Accuracy**        |           |        | **0.8135** | 7929    |
| **Macro avg**       | 0.73      | 0.69   | 0.66     | 7929    |
| **Weighted avg**    | 0.83      | 0.81   | 0.81     | 7929    |

The decision tree model trained to classify news popularity achieved an overall accuracy of 81.35%, a very good score considering the number of classes and the unbalanced nature of the dataset.
The "Slightly Popular," "Moderately Popular," and "Popular" classes are well classified, with f1-scores above 0.70 and high recall.
- The "Unpopular" class, although it has a small number of examples, is surprisingly well captured (recall 0.80), but with poor accuracy (0.33), indicating many false positives.
- The "Viral" class is more difficult to identify: it has good accuracy (0.84) but very low recall (0.24), suggesting that the model tends to be conservative and classify very few articles as viral — which may be acceptable if false alarms are to be avoided.
The model was trained with carefully chosen hyperparameters:
- max_depth=20 for increased flexibility,
- min_samples_leaf=2 to reduce overfitting,
- class_weight='balanced' to address class imbalance.

Thus, the model offers solid performance and is suitable as a basic solution for predicting news popularity. Extreme classes, such as "Viral" or "Unpopular," can be further improved through oversampling, ensemble methods, or more complex models such as Random Forest or MLP.

#### Random Forests

![alt text](images/news_popularity/algorithms/random_forest_news.png)

| Class                | Precision | Recall | F1-score | Support |
|---------------------|-----------|--------|----------|---------|
| Moderately Popular  | 0.87      | 0.80   | 0.83     | 2401    |
| Popular             | 0.71      | 0.71   | 0.71     | 1074    |
| Slightly Popular    | 0.87      | 0.92   | 0.90     | 3799    |
| Unpopular           | 0.33      | 0.80   | 0.47     | 218     |
| Viral               | 0.84      | 0.24   | 0.37     | 437     |
|                     |           |        |          |         |
| **Accuracy**        |           |        | **0.8135** | 7929    |
| **Macro avg**       | 0.73      | 0.69   | 0.66     | 7929    |
| **Weighted avg**    | 0.83      | 0.81   | 0.81     | 7929    |
Accuracy (Random Forest): 0.8135

The Random Forest model achieved an overall accuracy of 81.35% in classifying the popularity of news articles, providing results almost identical to the individual decision tree, but with added robustness and stability.
- Performance is very good for the "Slightly Popular," "Moderately Popular," and "Popular" classes, which represent the majority of the set. These have f1-scores above 0.70 and a very high recall.
- The "Unpopular" class is well captured in recall (0.80), but suffers in accuracy (0.33), suggesting a large number of false positive classifications.
- The "Viral" class again has identification difficulties, with a recall of only 0.24, although the precision is good (0.84), similar to Decision Tree.

The model was trained with the following hyperparameters:
- n_estimators=100: a forest of 100 trees,
- max_depth=20: to allow learning complex relationships,
- min_samples_leaf=2: to reduce overfitting on small leaves,
- criterion='entropy': to maximize information gain, - class_weight='balanced': to counterbalance the uneven distribution of classes.

Thus, Random Forest is a solid model for this task, offering a very good balance between accuracy and generalization. For underrepresented classes, performance can be improved by oversampling techniques, increasing the number of estimators, or adjusting class weights.

#### Logistic Regression

![alt text](images/news_popularity/algorithms/logistic_regression_news.png)

| Class                  | Precision | Recall | F1-score | Support |
|------------------------|-----------|--------|----------|---------|
| Moderately Popular     | 0.38      | 0.72   | 0.50     | 2401    |
| Popular                | 0.13      | 0.07   | 0.09     | 1074    |
| Slightly Popular       | 0.78      | 0.52   | 0.63     | 3799    |
| Unpopular              | 0.00      | 0.00   | 0.00     | 218     |
| Viral                  | 0.05      | 0.03   | 0.04     | 437     |
|                        |           |        |          |         |
| **Accuracy**           |           |        | **0.48** | 7929    |
| **Macro Avg**          | 0.27      | 0.27   |  0.25    | 7929    |
| **Weighted Avg**       | 0.51      | 0.48   | 0.46     | 7929    |

Accuracy (Manual Logistic Regression): 0.4777

For the problem of classifying news popularity, a multi-class logistic regression model was implemented using a One-vs-Rest (OvR) strategy. This involves training a binary logistic model for each possible class, modeling each "vs the rest."
Encoding: Categorical attributes were processed using One-Hot Encoding (only for those with low cardinality), using ColumnTransformer. Numerical attributes were kept unchanged.
Optimization:
- Algorithm: Standard Gradient Descent
- Learning rate: 0.1
- Number of epochs: 3000
- L2 regularization: λ = 0.01
Each binary model in OvR was trained independently, and at inference, the probability scores for each class were compared to choose the final prediction.
The model achieved an overall accuracy of 47.7%, significantly better than random classification, but with uneven performance:
- The "Moderately Popular" and "Slightly Popular" classes are the best classified (recall 72% and 52%)
- The "Popular" and "Viral" classes are often confused with the dominant classes
- The "Unpopular" class has accuracy and recall close to 0, suggesting systematic confusion with nearby classes

The distribution of results suggests that, although OvR allows the application of logistic regression to multi-class classification, limited linear separability and imbalance between classes strongly affect performance.

#### MLP

![alt text](images/news_popularity/algorithms/mlp_news.png)

| Class               | Precision | Recall | F1-score | Support |
|---------------------|-----------|--------|----------|---------|
| Moderately Popular  | 0.62      | 0.73   | 0.67     | 2401    |
| Popular             | 0.38      | 0.64   | 0.48     | 1074    |
| Slightly Popular    | 0.88      | 0.68   | 0.77     | 3799    |
| Unpopular           | 0.65      | 0.22   | 0.33     | 218     |
| Viral               | 0.17      | 0.13   | 0.15     | 437     |

An MLP (Multi-Layer Perceptron) model was trained using the scikit-learn library to predict the popularity category of news articles (popularity_category). The model was trained on the processed data (X_train, y_train) and used the following hyperparameters:
Architecture:
- Hidden layers (hidden_layer_sizes): two hidden layers with 256 and 128 neurons;
- Activation function (activation): ReLU, chosen for its efficiency in gradient propagation and non-linear learning.
Optimization:
- Algorithm (solver): adam, an efficient optimizer for large and sparse data;
- Learning rate (learning_rate_init): 0.001 — default value suitable for stability;
- Maximum number of epochs (max_iter): 3000 — allows the network sufficient training cycles for convergence.
Regularization:
- Early stopping: enabled (early_stopping=True) — training stops when validation performance no longer increases;
- L2 regularization (alpha): 0.0001 — prevents overfitting by penalizing large weights.

The model achieved an overall accuracy of 64.48%, with the following important observations extracted from the confusion matrix:
- Slightly Popular and Moderately Popular are the best classified (f1-scores of 0.77 and 0.67), being also the most frequent classes;
- Popular performs decently (f1 ≈ 0.48), but confusions with Moderately Popular are frequent;
- The Unpopular and Viral classes are poorly classified (f1 ≈ 0.15–0.33), with numerous confusions with Popular and Slightly Popular.
- The model tends to favor dominant classes and has difficulty learning clear representations for rare classes. We also note that Viral is frequently confused with Popular, and Unpopular is distributed among several classes.

![alt text](images/news_popularity/algorithms/mlp_loss_curve.png)

![alt text](images/news_popularity/algorithms/mlp_accuracy_curve.png)

## Comparison
| Method | Accuracy-AIR | Accuracy-NEWS |
|--------|--------------|----------------|
| dt     | 0.8886       | 0.8135         |
| rf     | 0.9111       | 0.8135         |
| lr     | 0.6953       | 0.4780         |
| mlp    | 0.9704       | 0.6102         |
