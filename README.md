#### dsc-5-capstone-project-online-ds-pt-112618

# Module 5 Capstone -Red and White Wine Predictions


## Overview

Can we predict red or white wine from a dataset?

How we’ll do this:
* Look at attributes (or features) of each type of wine
* Look at low, medium, and high quality wines statistical interactions
 
The capstone has been broken down into multiple Jupyter Notebooks.  The breakdown consists of the following Jupyter Notebooks:
<br>1. EDA - Red and White wine datasets
<br>2. EDA - Merging of Red and White wine datasets
<br>3. Statistical Analysis - Red and White wine datasets 
<br>4. Machine Learning - Merged dataset 
<br>4a. Machine Learning - Balance Merged dataset
<br>5. Deep Learning - Merged dataset
<br>5a. Deep Learning - Balance Merged dataset


## Problem Statement

The crux of this capstone is can machine learning and deep learning predict the type of wine (either red or white) from features found within the dataset?  

Methodsology for this analysis will consists of exploratory data analysis (EDA), and categorizing wine into categories and looking at statistical analyses of each group.

## Data Source Overview

I obtained my wine dataset from the University of California, Irvine (UCI), which consisted of a red and white wine dataset.  The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. For more details, consult: [Web Link] or the reference [Cortez et al., 2009].  

The dataset was obtained from the University of California, Irvine (UCI): https://archive.ics.uci.edu/ml/datasets/Wine+Quality

## Scrubbing the Data

There were twelve features (or columns) and the dataset between the two are imbalanced (1599 rows for red wine dataset, 4898 rows for white wine dataset). 

Looking solely at the red wine dataset, it consisted of 8 quality categories.  The quality feature, which was based on sensory data was a score between 0-10.  I was interested primarily in this feature since it would drive my analyses for this capstone project.  Additionally, null values were not present, and both datasets had matching features.  

## 1) Exploratory Data Analysis

Comparing quality against a few features, the red wine categories were in a normal, distribution. I did not notice any strong relationships with red wine quality and the dependent variables. However, I noticed as citric acid, alcohol and sulphates increase as red wine quality increase.  I did the same for white wine and there really are no clear trends as white wine quality increases. Like red wine, I believe a correlation matrix would better help to understand the relationship between features.

<img src="https://github.com/Sugaboo/dsc-5-capstone-project-online-ds-pt-112618/blob/master/Red%20wine%20countplot.png" alt="Alt text that describes the graphic" title="Red wine counterplot" />

<img src="https://github.com/Sugaboo/dsc-5-capstone-project-online-ds-pt-112618/blob/master/Citric%20acid%20countplot_red.png" alt="Alt text that describes the graphic" title="Citric acid countplot_red" />

<img src="https://github.com/Sugaboo/dsc-5-capstone-project-online-ds-pt-112618/blob/master/Sulphate%20countplot_red.png" alt="Alt text that describes the graphic" title="Sulphate countplot_red" />

<img src="https://github.com/Sugaboo/dsc-5-capstone-project-online-ds-pt-112618/blob/master/White%20wine%20countplot.png" alt="Alt text that describes the graphic" title="White wine countplot" />

<img src="https://github.com/Sugaboo/dsc-5-capstone-project-online-ds-pt-112618/blob/master/Citric%20acid%20countplot_white.png" alt="Alt text that describes the graphic" title="Citric acid countplot_white" />

<img src="https://github.com/Sugaboo/dsc-5-capstone-project-online-ds-pt-112618/blob/master/Sulphate%20countplot_white.png" alt="Alt text that describes the graphic" title="Sulphate countplot_white" />

## 2) EDA - Merging of Red and White wine datasets

I merged both datasets together and separated them into quality classes (i.e. – low, medium and high quality).  I was curious about the descriptive stats of each quality class, a noticed a few trends.  

The highlights from descriptive statistic of wine quality class were the following: mean total sulfur dioxide and residual sugar content seems to be much higher in white wines than in red wines; citric acid is more present in white wines, while fixed acidity, volatile acidity and sulphates are more present in red wine; and red wines have double concentration of chlorides then white wines.  

I compared dependent features to the independent feature (quality).  For instance, looking at fixed acidity against quality showed a fairly symmetric distribution with a long tail. The box plots showed medians for many quality categories are almost similar, along with multiple.  Overall, many features had a normal distribution which was skewed to the right and box plots of each varied.  

**Fixed acidity distplot**
<img src='https://github.com/Sugaboo/dsc-5-capstone-project-online-ds-pt-112618/blob/master/Fixed%20acidity%20distplot.png'>

**Fixed acidity boxplot**
<img src='https://github.com/Sugaboo/dsc-5-capstone-project-online-ds-pt-112618/blob/master/Fixed%20acidity%20boxplot.png'>

Moving onto the wine quality and wine quality classes correlation and pairplot in Seaborn, some of the following trends were observed with wine quality: 
* Total (70%) and free (47%) sulfur dioxides have the highest correlation with white wines.
* The volatile acid (-65%) and chlorides (-51%) have a negative correlation with color. This indicates a tendency to red wines classification.
* Density has a relatively high negative correlation to alcohol (-69%). This is confirmed by the decreasing linear trend from left to right. Density also has a relatively high positive correlation to residual sugar (55%), which is reinforced by two white wine outliners

## 3) Statistical Analysis - Red and White wine datasets

Multiple types of statistics were run in Jupyter Notebook.  For instance, density has a relatively high negative correlation with alcohol.  Moreover, total sulfur dioxide and residual sugar content seem to be much higher in white wines than in red wines.

**Residual sugar in red wine vs. density**
<img src='https://github.com/Sugaboo/dsc-5-capstone-project-online-ds-pt-112618/blob/master/PC_red.png'>

**Residual sugar in white wine vs. density**
<img src='https://github.com/Sugaboo/dsc-5-capstone-project-online-ds-pt-112618/blob/master/PC_white.png'>

**Alcohol percentage in red wine quality**
<img src='https://github.com/Sugaboo/dsc-5-capstone-project-online-ds-pt-112618/blob/master/OH_red.png'>

**Alcohol percentage in white wine quality**
<img src='https://github.com/Sugaboo/dsc-5-capstone-project-online-ds-pt-112618/blob/master/OH_white.png'>

## 4) Machine Learning - Merged dataset

Using an 80/20 train/test split, I focused on classification primarily for my predictive modeling.  Additionally, I distinguished red wines with a value of 1 and white wines with a value of 0.  My output and confusion matrices generated the following results: 
* Random Forest: 99%
* K-Nearest Neighbors (kNN): 94%
* Logistic Regression: 98%
* SVM: 94%
* Decision Tree: 96%

**Random Forest Confusion Matrix**
<img src='https://github.com/Sugaboo/dsc-5-capstone-project-online-ds-pt-112618/blob/master/Random%20Forest%20confusion%20matrix.PNG'>

## 4a) Machine Learning - Balance Merged dataset

The above result was ideal; however, there was a big problem with relying on my output...my dataset was not balanced!  So, I was skeptical of all the results, since they had such a high level of accuracy. 

Using the Up-Sample Minority Class technique to balance my data (documentation obtained from: https://elitedatascience.com/imbalanced-classes). Re-runing my predicitive modeling. Now I can see my red and white wine classes are balanced. 

Again Random Forest performed the highest at 99%. This time I feel confident this is an accurate result, since the dataset was balanced.

<img src='https://github.com/Sugaboo/dsc-5-capstone-project-online-ds-pt-112618/blob/master/Random%20Forest%20CM_balanced.png'>

## 5) Deep Learning - Merged dataset

I decided to use Keras for modeling my data.  After setting up the training and testing data (i.e. - compiling the metrics), I ran the model. The final output showed the model had 99% accuracy. So, it performed as well as the Random Forest machine learning model.

<img src='https://github.com/Sugaboo/dsc-5-capstone-project-online-ds-pt-112618/blob/master/Keras%20confusion%20matrix.PNG'>

However, similarly to the Machine Learning models, my dataset was unbalanced.  Thus, I needed to balanced the dataset to ensure that my output and results could be trusted.

## 5a) Deep Learning - Balance Merged dataset

I re-ran the Keras model after I used the Up-Sample Minority Class technique (please see documenation found in Jupter Notebook 4a). I ended up with a 98% level of accruacy!  Thus it performed around the same percentage as the unbalanced dataset.  However, since this analysis was done with a balanced dataset, I strongly believe this result.

<img src='https://github.com/Sugaboo/dsc-4-final-project-online-ds-pt-112618/blob/master/mod%204%20tumor%20schttps://github.com/Sugaboo/dsc-5-capstone-project-online-ds-pt-112618/blob/master/Keras%20CM_balanced.png'>

## Conclusions

* The EDA showed a lot of strong relationships between the wine type features.

* Our statistical analyses of red and white wines revealed positive correlations. 

* Imbalanced datasets: Random Forest had 99% accuracy, Keras had a 100% accuracy in predicting wine type.

* Balanced datasets: Random Forest had 99% accuracy, and Keras had 98% accuracy in predicting wine type.

* Wine type classes were imbalanced, which I believe influenced high levels of modeling accuracy.

## Future Work and Recommendations

* Improve parameters in Keras deep learning model to improve accuracy

* Possibly adjust training and testing set of data to see if that will yield higher accuracy

* Add more features to the dataset, making it more robust for testing

* Explore other deep learning models to determine level of accuracy
