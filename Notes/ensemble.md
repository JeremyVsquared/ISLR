Ensemble learning is a technique of combining multiple learners to create a stronger, more robust learner often by combining the output of these learners by voting, weighted voting, averagin, or perhaps even another model. This is commonly used as a technique to combat variance or bias present in a single model. 

Common examples of ensemble learning are _bagging_, _boosting_, and _stacking_.

# Bagging

_Bagging_ is an abbreviation for _Bootstrap Aggregation_. This is an ensemble practice of bootstrapping a dataset and training the same model separately on each of the datasets. When predicting, each of these model instances perform their prediction and the output is aggregated in some way, such as averaging in the case of a regression or voting in the case of classification. This method is especially useful when working with unstable models that tend toward high variance.

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

_Boosting_ is superficially similar to bagging, however individual models are trained on successively modified versions of the dataset and the output is weighted upon aggregation. The weighting is determined by the perfomance of a given model. The principle motivation is to train models for difficult to resolve inputs, focusing each on the cases where prior models failed, such that the resulting learner has a collection of specialty models from which to poll for output.

Popular examples of boosting algorithms are AdaBoost, Gradient Boosting and xgboost.


# References

- [The Elements of Statistical Learning](http://www-stat.stanford.edu/~tibs/ElemStatLearn/download.html)