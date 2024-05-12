An experiment was conducted on 5000 participants to study the effects of age and physical health on hearing loss, specifically the ability to hear high pitched tones. This data displays the result of the study in which participants were evaluated and scored for physical ability and then had to take an audio test (pass/no pass) which evaluated their ability to hear high frequencies. The age of the user was also noted. Is it possible to build a model that would predict someone's liklihood to hear the high frequency sound based solely on their features (age and physical score)?

Features

age - Age of participant in years
physical_score - Score achieved during physical exam
Label/Target

test_result - 0 if no pass, 1 if test passed





Coefficient Interpretation
Things to remember:

These coeffecients relate to the odds and can not be directly interpreted as in linear regression.
We trained on a scaled version of the data
It is much easier to understand and interpret the relationship between the coefficients than it is to interpret the coefficients relationship with the probability of the target/label class.



The odds ratio
For a continuous independent variable the odds ratio can be defined as:


For a continuous independent variable the odds ratio can be defined as:


This exponential relationship provides an interpretation for
𝛽1
The odds multiply by
𝑒𝛽1
for every 1-unit increase in x.



This means:

We can expect the odds of passing the test to decrease (the original coeff was negative) per unit increase of the age.
We can expect the odds of passing the test to increase (the original coeff was positive) per unit increase of the physical score.
Based on the ratios with each other, the physical_score indicator is a stronger predictor than age.























