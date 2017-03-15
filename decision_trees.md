# Trees

Tree based methods involve segmenting the data such that it is divided into numerous regions, also referred to as _leaves_ or _terminal nodes_, as defined by the rules developed by the algorithm. They tend to be easily interpreted but generally perform poorly when used for prediction without modification (ie, bagging, boosting, or ensemble methods). Decision trees divide the data based upon rules about a given observation, such as $x_i > 4.5$, and observations fall through these _splits_ or _internal nodes_ until reaching a leaf.

Decision trees can be applied to classification problems by applying the most common label at the predicted leaf or regression problems by averaging the target at the predicted leaf.

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

Very interpretable but suffer from poor predictive performance and high variance without futher complication, which makes it less interpretable. As such, decision trees are not popularly used as predictive tools, but enhanced versions such as bagged trees and random forests are generally considered to be more effective extensions.

# Hierarchical Mixtures of Experts (HME)

A _hierarchical mixture of experts_ (HME) are a tree based algorithm but unlike the standard decision tree, the splits are probabilistic, not binary, and a linear or logistic regression is fit at each leaf to generate the prediction. In the context of HME's, the leaf is referred to as an "expert" as it really is an independent predictor.