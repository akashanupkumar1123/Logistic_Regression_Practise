In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement on a company website. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.

This data set contains the following features:

'Daily Time Spent on Site': consumer time on site in minutes
'Age': cutomer age in years
'Area Income': Avg. Income of geographical area of consumer
'Daily Internet Usage': Avg. minutes a day consumer is on the internet
'Ad Topic Line': Headline of the advertisement
'City': City of consumer
'Male': Whether or not consumer was male
'Country': Country of consumer
'Timestamp': Time at which consumer clicked on Ad or closed window
'Clicked on Ad': 0 or 1 indicated clicking on Ad




Get the Data
Read in the advertising.csv file and set it to a data frame called ad_data.







Exploratory Data Analysis
Let's use seaborn to explore the data!

Try recreating the plots shown below!

** Create a histogram of the Age**


----
Create a jointplot showing Area Income versus Age.



-------
Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.


----------
** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**




----------
** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**



-----------

Logistic Regression
Now it's time to do a train test split, and train our model!

You'll have the freedom here to choose columns that you want to train on!

** Split the data into training set and testing set using train_test_split**



--------














