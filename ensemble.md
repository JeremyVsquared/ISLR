Ensemble learning is a collection of techniques which combine multiple learners to create a stronger, more robust learner often by combining the output of these learners by voting, weighted voting, averagin, or perhaps even another model. This is commonly used as a technique to combat variance or bias present in a single model. This tends to be a very effective tool as it functions as a "committee" of learners, rather than relying entirely upon a single hypothesis to wholly and accurately model the reality of some data.

Common examples of ensemble learning are _bagging_, _boosting_, and _stacking_.

# Bagging

_Bagging_ is an abbreviation for _Bootstrap Aggregation_. This is an ensemble practice of bootstrapping a dataset such that a model can be independently trained on each of the resampled datasets. When predicting, each of these model instances perform their prediction and the output is aggregated in some way, such as averaging in the case of a regression or voting in the case of classification. This method is especially useful when working with unstable models that tend toward high variance or data with a lot of noise.

```{python}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("...")
X_train, X_test, y_train, y_test = train_test_split(df, test_size=0.25)

knn = KNeighborsClassifier()
bgc = BaggingClassifier(knn, n_estimators=5, max_samples=0.5, max_features=0.5)

bgc.fit(X_train, y_train)
y_pred = bgc.predict(X_test)
```

# Boosting

_Boosting_ is superficially similar to bagging, however individual models are trained on successively modified versions of the dataset and the output is weighted upon aggregation. The weighting is determined by the perfomance of a given model. The principle motivation is to train models for difficult to resolve inputs, focusing each on the cases where prior models failed, such that the resulting learner has a collection of specialty models from which to poll for output. Boosting is very resistant to overfitting and, when working with a clean dataset with little noise,  is generally more accurate than bagging.

Popular examples of boosting algorithms are AdaBoost, Gradient Boosting and xgboost.

# Stacking

In practice, _stacking_ is performed in two phases. The first phase is generating a collection of models which are trained on the base input data. This collection of models each provide their output, along with the real value of the original data set, to another data set. This second data set is used to train a final model, the output of which is the true output of the predictive system. _Stacking_ is a form of _meta-learning_ in that it is learning from engineered, derived output rather than the base data alone. 

# References

- [The Elements of Statistical Learning](http://www-stat.stanford.edu/~tibs/ElemStatLearn/download.html)
- [An Experimental Comparison of Three Methods for Constructing Ensembles of Decision Trees: Bagging, Boosting, and Randomization](http://link.springer.com/article/10.1023/A%3A1007607513941)
- [Machine Learning A Probabilistic Perspective](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020)