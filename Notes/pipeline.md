# Pipeline

Pipeline of PCA and Decision Tree Regressor

```{python}
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