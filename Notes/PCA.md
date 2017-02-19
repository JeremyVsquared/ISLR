*Principal components* provide a reduced dimensional representation of data. In scenarios where the number of features $p$ is very large, it can, in some cases, be prohibitively difficult to visualize and compute the relationships between the features or between the features and the output. In these cases, it is helpful to reduce the number of features being considered at any given time. Principal components provides us guidance on constructing and selecting representations of the data that will still be informative. In brief, this is accomplished by reducing dimensions along the lines of the greatest variance within the data, thus reducing a large dimensional dataset to a desired number of dimensions.

# Principal Components Analysis

Consider the scenario of encountering a new data set with 250 features. This feature count $p$ is too large to quickly or easily visualize in a collection of scatter plots and would likely not lead to a good fit if it were to be thrown at a random selection of predictive algorithms. Additionally, when encountering data sets with this many features in practice, it would be unusual for every feature to actually be an important factor in calculating a relationship between these predictors and the outcome. Thus the dataset would become much easier to understand and visualize if the number of dimensions to reduced to a more manageable size.

We need a way to determine which variables are important in representing the overall dataset without needing an ellimination process of trial and error, working through every permutation of the feature set. Principal components analysis provides us this.

Python

```{python}
from sklearn.decomposition import PCA

pca = PCA(n_components=2, svd_solver='full')
pca.fit(X)              
print(pca.explained_variance_ratio_)
```

R

```{r}
# center & scale. parameters standardize the variables prior to decomposition
pca = prcomp(df, center=TRUE, scale.=TRUE)
summary(pca)
```

# Principal Components Regression

Principal components can be used to construct a regression model wherein the principal components of a data set are used as predictors rather than the whole, original set of features. This can be very helpful in cases where (1) the entire set of features is sufficiently large to become prohibitively computationally intensive or (2) the entire feature set is providing a poor fit due to a great deal of inter-predictor variance.

```{r}
require(pls)

pcr.mdl = pcr(y~., data=df, scale=TRUE, validation="CV)
summary(pcr.mdl)
pred = predict(pcr.mdl, test, ncomp=3)
```