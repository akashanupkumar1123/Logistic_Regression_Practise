Logistic regression is the go-to linear classification algorithm for two-class problems. It is easy to implement, easy to understand and gets great results on a wide variety of problems, even when the expectations the method has for your data are violated.




Description
Logistic Regression
Logistic regression is named for the function used at the core of the method, the logistic function.

The logistic function, also called the Sigmoid function was developed by statisticians to describe properties of population growth in ecology, rising quickly and maxing out at the carrying capacity of the environment. It’s an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1, but never exactly at those limits.

11+𝑒−𝑥
 
𝑒  is the base of the natural logarithms and  𝑥  is value that you want to transform via the logistic function.




The logistic regression equation has a very simiar representation like linear regression. The difference is that the output value being modelled is binary in nature.

𝑦̂ =𝑒𝛽0+𝛽1𝑥11+𝛽0+𝛽1𝑥1
 
or

𝑦̂ =1.01.0+𝑒−𝛽0−𝛽1𝑥1
 
𝛽0  is the intecept term

𝛽1  is the coefficient for  𝑥1 
𝑦̂   is the predicted output with real value between 0 and 1. To convert this to binary output of 0 or 1, this would either need to be rounded to an integer value or a cutoff point be provided to specify the class segregation point.






Making Predictions with Logistic Regression
𝑦̂ =1.01.0+𝑒−𝛽0−𝛽1𝑥𝑖
 
𝛽0  is the intecept term

𝛽1  is the coefficient for  𝑥𝑖 
𝑦̂   is the predicted output with real value between 0 and 1. To convert this to binary output of 0 or 1, this would either need to be rounded to an integer value or a cutoff point be provided to specify the class segregation point.




Learning the Logistic Regression Model
The coefficients (Beta values b) of the logistic regression algorithm must be estimated from your training data.

Generally done using maximum-likelihood estimation.

Maximum-likelihood estimation is a common learning algorithm

Note the underlying assumptions about the distribution of your data

The best coefficients would result in a model that would predict a value very close to 1 (e.g. male) for the default class and a value very close to 0 (e.g. female) for the other class.

The intuition for maximum-likelihood for logistic regression is that a search procedure seeks values for the coefficients (Beta values) that minimize the error in the probabilities predicted by the model to those in the data.




Learning with Stochastic Gradient Descent
Logistic Regression uses gradient descent to update the coefficients.

Each gradient descent iteration, the coefficients are updated using the equation:

𝛽=𝛽+learning rate×(𝑦−𝑦̂ )×𝑦̂ ×(1−𝑦̂ )×𝑥






















