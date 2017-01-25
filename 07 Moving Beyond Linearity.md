# 7 Moving Beyond Linearity

Standard linear regression can have significant limitations in terms of predictive power because it is fundamentally built upon an assumption of a linear relationship, which is not always realistic. The following are nonlinear methods which may provide more accurate predictions:

1. _Polynomial regression_: extends the linear model with exponentially raised predictors; ie, a linear model of the form $x_1 + x_2 + \epsilon$ could become $x_1 + x^2_1 + x_2 + x^2_2 + x^3_2 + \epsilon$
2. _Step functions_: cut the range of a variable in $K$ distinct regions in order to produce a qualitative variable
3. _Regression splines_: an extension of polynomial and step functions; involve dividing the range of $X$ into $K$ distinct regions, within each a polynomial function is fit but are constrained so that they join smoothly at the region boundaries, as known as _knots_
4. _Smoothing splines_: result from minimizing a residual sum of squares criterion subject to a smoothness penalty
5. _Logical regression_: similar to splines but the regions are allowed to overlap
6. _Generalized additive models_: allow us to extend the methods above to deal with multiple predictors

___

## Polynomial Regression

Here we would replace the standard formula

$y_i=\beta_0 + \beta_1x_1 + \epsilon_i$

with 

$y_i = \beta_0 + \beta_1x_1 + \beta_2x_i^2 + \beta_3x^3_i + ... + \beta_dx^d_i + \epsilon_i$

For large enough $d$, a polynomial regression allows us to produce an extremely nonlinear curve. In practice, it is unusual to use $d$ greater than 3 or 4 as it can produce some overly flexible, erratic fits.

___

## Step Functions

Using polynomial functions of the features imposes a structure on the nonlinear function of $X$. We can instead use step functions to avoid this. Practically, this is done by dividing $X$ into _bins_ and fitting a different constant in each bin. This amounts to converting a continuous variable into an _ordered categorical variable_.

In other words, we create a series of boundaries $c_1, c_2, ..., c_K$ in the range of $X$ and then construct $K+1$ new variables

$c_0(x) = I(x < c_1)$
$c_1(x) = I(c_1 \leq x \leq c_2)$
$...$
$c_{K-1}(x) = I(c_{K-1} \leq x \leq c_K)$
$c_K(x) = I(c_K \leq x)$

Where $I()$ is an _indicator function_ that returns 1 if the condition is true, 0 otherwise, creating our dummy variables. We then use least squares to fit a linear model using $C_1(x), C_2(x), ..., C_K(x)$ as predictors

$y_i = \beta_0 + \beta_1C_1(x_i) + \beta_2C_2(x_i) + ... + \beta_KC_K(x_i) + \epsilon_i$

because this method works in a discrete fashion, it can miss the momentum, or direction, in changing values.

___

## Basis Functions

Polynomial and step functions are special cases of basis functions. The concept of basis functions is to have a family of functions or transformations to be applied to a variable $X:b_1(X), b_2(X), ..., b_K(X)$. Instead of fitting a linear model in $X$, we fit the model

$y_i = \beta_0 + \beta_1b_1(x_i) + \beta_2b_2(x_i) + ... + \beta_Kb_k(x_i) + \epsilon_i$

In the case of polynomial regression, the basis functions are $b_j(x_i) = x^j_i$, and for step functions the basis function is $b_j(x_i) = I(c_j \leq x_i < c_{j+1})$. This means that all of the inference the inference tools for linear models, such as standard errors for the coefficient estimates and F-statistics for the model's overall significance, are equally applicable here.

___

## Regression Splines

A flexible class of basis functions

### Piecewise Polynomials

Instead of fitting a high-degree polynomial over the entire range of $X$, piecewise polynomial regression involves fitting separate low-degree polynomials over different regions of $X$. The boundaries in $X$ where the $\beta$'s change are called _knots_.

For example, a piecewise cubic polynomial with only 1 knot at a point _c_ takes the form

$y_i = \{^{\beta_{01} + \beta_{11}x_i + \beta_{21}x^2_i + \beta_{31}x^3_i + \epsilon_i\ \ \ if\  x_i < c}_{\beta{02} + \beta_{12}x_i + \beta_{22}x^2_i + \beta_{32}x^3_i + \epsilon_i\ \ \ if\  x_i \geq c}$

In other words, a cubic polynomial is fit to a subset of $X$ where $x_i < c$, and these coefficients are $\beta_{i1}$, and another cubic polynomial is fit to a subset $X$ where $x_i \geq c$ and these coefficients are $\beta_{i2}$.

One potential downside of piecewise polynomial fits is that they can be discontinuous. That is to say that the predictions can jump between values at the knots if the predictions do not align, which is not at all encouraged by the algorithm.

### Constraints and Splines

It is possible to remedy the discontinuous nature of piecewise polynomials by fitting under the _constraint_ that the fitted curve must be continuous. The spline is a reference to enforcing the continuity.

### The Spline Basis Representation

We can use the basis model to represent a regression spline. A cubic spline with $K$ knots can be modeled as 

$y_i = \beta_0 + \beta_1b_1(x_i) + \beta_2b_2(x_i) + ... + \beta_{K+3}b_{K+3}(x_i) + \epsilon_i$

for an appropriate choice of basis functions $b_1, b_2, ..., b_{K+3}$. The model can then be fit using least squares.

In other words, in order to fit a cubic spline to a data set with $K$ knots, we perform least squares regression with an intercept and $3+K$ predictors, of the form $X, X^2, X^3, h(X, \xi_1), h(X, \xi_2), ..., h(X, \xi_K)$, where $\xi_1, ..., \xi_K$ are the knots. This amounts to estimating a total of $K+4$ regression coefficients; for this reason, fitting a cubic spline with $K$ knots uses $K+4$ degrees of freedom.

The most direct way to represent a cubic spline is to start with a basis for a cubic polynomial $(x, x^2, x^3)$ and then add one _truncated power basis_ function per knot. A truncated power basis function is defined as

$h(x, \xi) = (x - \xi)^3_t = \{^{(x-\xi)^3\ \ \ \ if\ x > \xi}_{0\ \ \ \ \ \ \ \ \ \ \ \ otherwise}$

where $\xi$ is the knot.

A _natural spline_ is a regression spline with additional _boundary constraints_: the function is required to be linear at the boundary, in the region where $X$ is smaller than the smallest knot or larger than the largest knot. This means that natural splines generally produce more stable estimates at the boundaries.

### Choosing the Number and Locations of the Knots

The regression spline is most flexible in regions that contain a lot of knots, because in those regions the polynomial coefficients can change rapidly. In practice, it is common to place knots in a uniform fashion.

An objective method for selecting the number of knots is to use cross-validation. With this method, we could fit the data with a chosen number of knots and evaluate the error, then repeat for differnet numbers of knots $K$. Then the value of $K$ giving the lowest $RSS$ is chosen.

### Comparison to Polynomial Regression

Regression splines often give superior results to polynomial regression because splines can introduce flexibility by increasing the number of knots but keeping the degree fixed.

___

## Smoothing Splines

### An Overview of Smoothing Splines

In fitting a smooth curve to data, what we want is to minimize $RSS$ where

$RSS = \sum^n_{i=1}(y_i - g(x_i))^2$

If $g(x_i)$ is unconstrained, the function will interpolate $y_i$, leading to overfitting and a jagged, non-smooth curve. We can avoid this by adding a tuning parameter term to the formula

$\sum^n_{i=1}(y_i - g(x_i))^2 + \lambda \int g'(t)^2dt$

The summation term in the above formula is a _loss function_ that encourages $g()$ to fit the data well, and the integral term is a _penalty term_ that works to suppress the variability of $g()$.

The notation $g'(t)$ indicates the second derivative of the function $g()$. The first derivative $g'(t)$ measures the slope of a function at $t$, and the second derivative corresponds to the amount by which the slope is changing. Hence, broadly speaking, the second derivative of a function is a measure of its _roughness_: it is large in absolute value if $g(t)$ is very wiggly near $t$, and it is close to 0 otherwise. The second derivative of a straight line is 0. The $\int$ notation is an integral, which we can think of as a summation over the range of $t$. In other words, $\int g'(t)^2dt$ is simply a measure of the total change in the function $g'(t)$, over its entire range. If $g()$ is very smooth, then $g'(t)$ will be close to constant and $\int g'(t)^2dt$ will take on a small value. The larger the value of $\lambda$, the smoother $g()$ will be.

### Choosing the Smoothing Parameter $\lambda$

The tuning parameter $\lambda$ controls the roughness of the spline, and hence the _effective degrees of freedom_. It is possible to show that as $\lambda$ increases from 0 to $\infty$, the effective degrees of freedom, which we write $df_\lambda$, decrease from $n$ to 2. $df_\lambda$ is a measure of the flexibility of the smoothing spline - the higher it is, the more flexible (and the lower-bias but higher-variance) the smoothing spline. Leave-one-out cross-validation error (LOOCV) can be computed very efficiently for smoothing splines with essentially the same cost as computing a single fit using the following

$RSS_{CV}(\lambda) = \sum^n_{i=1}(y_i - \hat{g}_\lambda^{(-i)}(x_i))^2 = \sum^n_{i=1}[\frac{y_i - \hat{g}_\lambda(x_i)}{1 - \{S_\lambda\}_i}]$

The notation $\hat{g}_\lambda^{(-i)}(x_i)$ indicates the fitted value for this smoothing spline evaluated at $x_i$, where the fit uses all of the training observations except for the $i^{th}$ observation $(x_i, y_i)$. In contrast, $\hat{g}_\lambda(x_i)$ indicates the smoothing spline function fit to all of the training observations and evaluated at $x_i$.

___

## Local Regression

Local regression is a different approach for fitting flexible non-linear functions, which involves computing the fit at a target point $x_0$ using only the nearby training observations.

Algorithm for local regression at $X=x_0$

1. Gather the fraction $x=\frac{k}{n}$ of training points whose $x_i$ are closest to $x$
2. Assign a weight $K_{i0} = K(x_i, x_0)$ to each point in this neighborhood, such that the point furthest from $x_0$ has weight 0, and the closest has the highest weight. All but these $k$ nearest neighbors get weight 0.
3. Fit a _weighted least squares regression_ of the $y_i$ on the $x_i$ using the aforementioned weights, by finding $\hat{\beta_0}$ and $\hat{\beta_1}$ that minimize $\sum^n_{i=1}K_{i0}(y_i - \beta_0 - \beta_1x_i)^2$
4. The fitted value at $x_0$ is given by $\hat{f}(x_0) = \hat{\beta}_0 + \hat{\beta}_1x_0

Note that the weights $K_{i0}$ will differ for each value of $x_0$. There are a number of choices to be made, such as how to define the weighting function $K$, and whether to fit a linear constant, or quadratic regression in #3, but the most important is the choice of _spans_ in #1. The smaller the choice of _s_, the more local and wiggly the fit; alternatively, a very large value of _s_ will lead to a global fit using all of the training observations. In this way, _s_ acts like the tuning parameter $\lambda$ we have seen in other algorithms.

One very useful generalization of local regression involves fitting a model that is global in some variables, but local in another, such as time. Such _varying coefficient_ models are a useful way of adapting a model to the most recently gathered data. Another generalization is fitting mdoels on mulitple variables, but local regression can perform poorly if $p$ is much larger than about 3 or 4 because there will generally be very few training observations close to $x_0$ in higher dimensions.

___

## Generalized Additive Models

The prior algorithms have all been extensions of simple linear regression. Now we'll discuss extensions to multiple linear rergession.

_Generalized additive models_ (GAMs) provide a general framework for extending a standard linear model by allowing non-linear functions of each of the variables, while maintaining additivity.

### GAMs for Regression Problems

A natural way to extend multiple linear regresison model in order to allow for non-linear relationships between each feature and the response is to replace each linear component $\beta_jx_{ij}$ with a (smooth) non-linear function $f_j(x_{ij})$. We would then write the model as

$y_i = \beta_0 + \sum^p_{j=1}f_j(x_{ij}) + \epsilon_i$

It is called an _additive_ model because we calculate a separate $f_j$ for each $X_j$, and then add together all of their contributions.