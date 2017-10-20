# Support Vector Machines

The motivation of Support Vector Machines is to find the optimal separating hyperplane with maximal margin for the training data.

Support vector machines can be applied to regression (_Support Vector Regressor_) or classification (_Support Vector Classifier_). For classification, it is important to note that SVMs are a binary classifier, which is to say that it can only distinguish two classes. In order to extend SVMs to identify more than two classes, it is necessray to implement multiple SVMs in one of two ways:

1. _One-vs-one_ or _all-pairs_, in which there are $(^k_2)$ SVMs and the given input is classified by the label it is most frequently assigned, or
2. _One-vs-all_ in which there are $k$ SVMs, each of which distinguishes between it's own class label and all other class labels.

The implementation of SVM involves two variable parameters used to optimize the algorithm. The first is $C$, which is a nonnegative tuning parameter that determines the degree to which the margin can be violated by the training observations. This effectively serves as a definition of permitted error. Secondly, there is the $\epsilon$ term which basically defines a margin of error which will be ignored by the algorithm in refining the position of the hyperplane.

```python
from sklearn.svm import SVR

svr = SVR(C=1.2, epsilon=0.2)
svr.fit(X_train, y_train)

y_pred = svr.predict(X_test)
```

```r
svm.model = svm(y ~., data=df_train)
y.pred = predict(svm.model, df_test)
```

## Strengths & Weaknesses

SVMs performs quite well modeling quadratic functions, rational differences, and ratios. It performs quite poorly on rational polynomials and differences.

## References

- [An Empirical Analysis fo Feature Engineering for Predictive Modeling](https://arxiv.org/pdf/1701.07852.pdf)