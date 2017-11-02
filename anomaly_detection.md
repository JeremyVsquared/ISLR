__Anomaly detection__ refers to discovering nonconforming data within a larger data set. Any patterns falling outside of expected values are referred to as __anomalies__, __outliers__, __exceptions__ or __dicordant observations__. In two dimenional data sets, these anomalies can be easily found visually by graphing the data, but in higher dimensions it can prove rather difficult and thus numerous algorithms have been proven or developed to effectively deal with this issue.

# Techniques 

## Rules based approaches

The most simple approach to finding outliers within data is what is known as __rule based__ detection. As the name implies, this method is the process of developing a set of rules by which anomalies can be identified. For example, if one is attempting to identify anomalies in weather patterns, an example rule may be temperature exceeding 105 degrees, rainfall exceeding 24 inches, or winds exceeding 25 MPH. These rules would then be applied to a dataset and observations meeting these criteria would be identified as anomalous. This method can be very effective when training data and domain expertise is available. While simple and easily explained to stakeholders, it does have a tendency to become quite brittle. Additionally, in the case of even a minimally complex data set, a fully developed rule set will become very complicated to maintain and extremely subjective.

## Clustering approaches

Beyond rule based anomaly detection, __clustering__ algorithms have also proven quite useful. These unsupervised methods are effective because on the whole anomalous observations will tend to fall spatially further away from expected observations. This will surface varying centroids that can be used to define expected and anomalous behavior in most situations, thus allowing the labeling of the anomalous observations by spatial division. The k-means algorithm is the most popular choice of clusering algorithms.

```python
# clustering anomaly detection code example 
```

## Density based approaches

When labeled data is available, __density-based__ algorithms such as kNN or local outlier factor can also be used. The assumption of the behavioral locality is the same here as it is when using a clustering algorithm, but the context of a supervised learning algorithm allows for greater validity as we can verify the results prior to deploying against new data in a live environment.

```python
# knn anomaly detection code example
```

## Types of anomalies

There are three broad categories of anomalies. First are __point anomalies__ are individual nonconforming data points. Second, a data point that is anomalous within certain contexts, but not others, are known as __contextual anomalies__. This could be an atypically high temperature reading during winter that would be normal during the summer. When the data becomes anomalous as a series but not as an individual observation, it becomes the third category which is known as __collective anomalies__. An example of this is commonly found in identity theft detection where a particular series of actions may raise an alert. Any given instance of anomalous behavior may be ignored, but a series of consecutive behaviors that fall outside of expected behavior becomes suspicious.

## Detecting Various Types

By definition, anomalies are rare and infrequent. As such, extreme class imbalance is very often present in labeled training data. This can be overcome by the standard class imbalance techniques, but there is an additional method that can be used here wherein two imbalanced datasets are developed, one for each conforming and one for nonconforming data. The ratio should be around 80% dominant class, 20% subordinate class, and separate classifiers are trained on each dataset. Then these classifiers are combined as an ensemble predictor which outputs the strongest predictor between the two original classifiers. It is important to evaluate the autocorrelation between the features as a recurrent neural network or time series classifier will be needed to address an autocorrelated data set.

When a labeled data set is not available, semi-supervised or unsupervised approaches will be necessary. These are less reliable than supervised techniques as they cannot be reliably evaluated prior to deployment, but sometimes this is unavoidable. Point anomalies are the easiest to detect in these cases. For instance, if 99% of transactions in a financial record are for values between $25 and $50, then any transaction for $1 or $100 would immediately raise suspicion. In these cases, it is often better to use moving averages rather than static values as they prove to be more resilient against noise within the data and can account for trends.

Detecting collective outliers without labeled training poses a greater challenge than point anomalies. The simplest case of this is a __univariate collective outlier__, such as a time series with a single feature. In this scenario, a collective outlier would arise when an unexpected series of values are detected. One simple way to find these would be to train a relatively simple model and look for large residuals. The assumption here is that a predicted value is a value expected to be found based upon the observations across the entire data set. Thus, when it differs greatly from the true value, it is due to factors outside of, and not explained by, expected behavior within the data and is anomalous. A more complicated method is to use a markov chain. Markov chains functionally evaluate the probability of a given sequence occurring and can used to effectively measure the likelihood of observing a chain of events given the patterns throughout the data set. This allows us to surface the anomalous sequences as they will be the least probable.

When the data becomes more complicated, including more than a single feature, we are the searching for __multivariate collective outliers__. This category requires special attention as the approaches differ based upon whether the data is __ordered__ or __unordered__.

