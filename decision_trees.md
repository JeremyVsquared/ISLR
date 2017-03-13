# Decision Trees

Tree based methods involve segmenting the data such that it is divided into numerous regions, also referred to as _leaves_ or _terminal nodes_, as defined by the rules developed by the algorithm. They tend to be easily interpreted but generally perform poorly when used for prediction without modification (ie, bagging, boosting, or ensemble methods). Decision trees divide the data based upon rules about a given observation, such as $x_i > 4.5$, and observations fall through these _splits_ or _internal nodes_ until reaching a leaf.

Decision trees can be applied to classification or regression problems.

```{python}
from sklearn.tree import DecisionTreeClassifier()

tree_mdl = DecisionTreeClassifier()
tree_mdl.fit(X_train, y_train)

y_pred = tree_mdl.predict(X_test)
```

```{r}
library(tree)

tree.mdl = tree(y~., data=df.train)
summary(tree.mdl)

y.pred = predict(tree.mdl, df.test)
```

## Strengths & Weaknesses

Very interpretable but poor predictive performance without futher complication, which makes it less interpretable.
