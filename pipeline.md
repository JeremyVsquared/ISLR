# Pipeline

Pipelines are chains of algorithms, feeding data from one to the other with the end goal of creating a more powerful learner. The purpose of this can be chain together predictors, performing a regression the output of which is fed through a classifier for instance. What can also be done is fixing a feature selection or dimensionality reduction algorithm in front of a learning algorithm.

Pipeline of PCA and Decision Tree Regressor

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

df = pd.read_csv("...csv")

pca = PCA()

dtr = DecisionTreeRegressor()

pipe = Pipeline(steps=[('pca', pca), ('dtr', dtr)])
pipe.set_params(pca__n_components=10, dtr__random_state=0, dtr__min_samples_split=8)

pipe.fit(df.drop('y', axis=1), df['y'])

pred = pipe.predict(df.drop('y', axis=1))
```

Pipeline of ANOVA and SVM

```python
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression

anova_filter = SelectKBest(f_regression, k=3)

clf = svm.SVC(kernel='linear')

anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(X, y)
anova_svm.predict(X)
```