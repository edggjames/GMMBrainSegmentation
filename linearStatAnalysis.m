% Student Number = 14062340
% Used by script StatAnalysis.m - Step 3

function [m, c, r, p] = linearStatAnalysis(X_data,Y_data)
% Assuming a linear relationship between the two sets of input data, this
% function calculates:

% 1) The linear regression coefficients in the form of y = mx + c,
% 2) The R^2 value - coefficient of determination - between 0 and 1,
% 3) The p-value level of statistical significance - based on t statistics.

% 1) Compute linear regression coefficients
% m is the gradient and c is the y-intercept
N = length(X_data);
X = ones(N,2);
X(:,2) = X_data;
Y = Y_data;
[coefficients] = X\Y;
c = coefficients(1);
m = coefficients(2);

% 2) Calculate r value - pearson linear correlation coefficient - between -1 and 1
term_1 = sum(X_data.*Y_data);
term_2 = length(X_data)*mean(X_data)*mean(Y_data);
term_3 = (N-1)*std(X_data)*std(Y_data);
r = (term_1 - term_2) / term_3;
% Pearson's r can range from -1 to 1. An r of -1 indicates a perfect negative 
% linear relationship between variables, an r of 0 indicates no linear relationship
% between variables, and an r of 1 indicates a perfect positive linear
% relationship between variables.

% 3) calculate t test statistic
t = r*sqrt((N-2)/(1-r^2));
% calculate p value based on two tailed test using student's t 
% cumulative distribution function
p = 2*tcdf(-abs(t), N-2);
% i.e this returns the cumulative distribution function (cdf) of the 
% Student's t distribution at the value -abs(t) using N-2 degrees of freedom.

% Using a signficance level of 5%, if this value is less than 0.05,
% then the relationship is statistically significant. 

end

