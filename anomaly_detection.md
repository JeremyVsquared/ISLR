__Anomaly detection__ refers to discovering nonconforming data within a larger data set. Any patterns falling outside of expected values are referred to as __anomalies__, __outliers__, __exceptions__ or __dicordant observations__. In two dimenional data sets, these anomalies can be easily found visually by graphing the data, but in higher dimensions or extremely large data sets it can prove rather difficult and thus numerous algorithms have been proven or developed to effectively deal with this issue. Anomaly detection is quite useful for such applications as component failure analysis, network intrusion, fraud detection and identity theft.

# Techniques

## Rules based approaches

The most simple approach to finding outliers within data is what is known as __rule based__ detection. As the name implies, this method is the process of developing a set of rules by which anomalies can be identified. For example, if one is attempting to identify anomalies in weather patterns, an example rule may be temperature exceeding 105 degrees, rainfall exceeding 24 inches, or winds exceeding 25 MPH. These rules would then be applied to a dataset and observations meeting these criteria would be identified as anomalous. This method can be very effective when training data and domain expertise is available. While simple and easily explained to stakeholders, it does have a tendency to become quite brittle and, in the case of even a minimally complex data set, will become very difficult to maintain and highly subjective.

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

# Challenges

Successful anomaly detection can prove quite challenging for a variety of reasons. In some cases, these challenges are adversarial in nature as the anomalies could prove to be the result of malicious behavior. In most cases, individual attempting to deliberately and systematically compromise a system will not make these attempts easy to to thwart and thus will alter their attack vectors over time with the hopes of increasing their success rate. This can make reliably detecting this within a data set rather difficult as the specifics of any given anomaly may change from one observation to another. Even when the anomalies are not deliberate or malicious, human behavior naturally changes over time, thus requiring adaptive methods in order to detect these changes as a "new normal" rather than anomalous.

It should be self-evident by considering a variety of scenarios in which anomaly detection could be employed that defining "normal" and "anomalous" can prove to be quite difficult, and the boundary between the two often proves to be rather imprecise. The very concept of anomalous varies wildly from one domain to another, thus applying a given algorithm or strategy is not always a simple matter. Additionally, most applications of anomaly detection involve data that are inherently noisy or unlabeled, and dealing with that presents challenges wholly apart from the anomaly detection itself. Labeling is a process that often requires manual intervention which is in most cases slow, time consuming, and thus expensive.

Methods broadly fall into the same general categories as machine learning algorithms: __supervised__, __semi-supervised__, and __unsupervied__. Here supervised anomaly detection involves labeled data for training and testing purposes. Semi-supervised anomaly detection involves training data that may have partially labeled data, or labels for only the normal class. These cases are typically approached by the building of a model trained to identify the normal class which is used to identify and label the anomalies. Unsupervised anomaly detection involves unlabeled data where the normal and anomalous observations are not identified.

# Types of Anomalies

There are three broad categories of anomalies. First are individual nonconforming data points known as __point anomalies__. These anomalies would be considered to be anomalous in and of themselves when compared to the rest of the dataset. An example of a point anomaly would be a record of rainfall with an average value of 12 inches per week with a single week of 28 inches. Second, a data point that is anomalous within certain contexts, but not others, are known as __contextual anomalies__. This could be an atypically high temperature reading during winter that would be normal during the summer. These most often arise in spatial or time series data. When the data becomes anomalous as a series but not as an individual observation, it is representative of the third category known as __collective anomalies__. An example of this is commonly found in identity theft detection where a particular series of actions may raise an alert. Any given instance of anomalous behavior may be ignored, but a series of consecutive behaviors that fall outside of expected behavior becomes suspicious.

## Detecting Various Types of Anomalies

By definition, anomalies are rare and infrequent. As such, extreme class imbalance is very often present in labeled training data. This can be overcome by the standard class imbalance techniques, but there is an additional method that can be used here wherein two imbalanced datasets are developed, one for conforming and one for nonconforming data. The ratio should be around 80% dominant class, 20% subordinate class, and separate classifiers are trained on each dataset. Then these classifiers are combined as an ensemble predictor which outputs the strongest predictor output of the two original classifiers. It is important to evaluate the autocorrelation between the features as a recurrent neural network or time series classifier will be needed to address an autocorrelated data set.

When a labeled data set is not available, semi-supervised or unsupervised approaches will be necessary. These are less reliable than supervised techniques as they cannot be reliably evaluated prior to deployment, but sometimes this is unavoidable. Point anomalies are the easiest to detect in these cases. For instance, if 99% of transactions in a financial record are for values between $25 and $50, then any transaction for $1 or $100 would be outliers. In these cases, it is often better to use moving averages rather than static values as they prove to be more resilient against noise within the data and can account for trends or seasonality.

### Detecting Univariate Collective Anomalies

Detecting collective outliers without labeled training poses a greater challenge than point anomalies. The simplest case of this is a __univariate collective anomalies__, such as a time series with a single feature. In this scenario, a collective anomaly would arise when an unexpected series of values are detected. One simple way to find these would be to train a relatively simple model and look for large residuals. The assumption here is that a predicted value is a value expected to be found based upon the observations across the entire data set. Thus, when it differs greatly from the true value, it is due to factors outside of, and not explained by, expected behavior within the data and is anomalous. A more complicated method is to use a markov chain. Markov chains functionally evaluate the probability of a given sequence occurring and can used to effectively measure the likelihood of observing a chain of events given the patterns throughout the data set. This allows us to surface the anomalous sequences as they will be the least probable.

### Detecting Multivariate Collective Anomalies

When the data becomes more complicated, including more than a single feature, we are the searching for __multivariate collective anomalies__. This category requires special attention as the approaches differ based upon whether the data is __ordered__ or __unordered__.

When the data is ordered, there are a variety of effective approaches. __Graph analysis__ proves useful as analyzing the graph flow will often make anomalies more obvious. For example, consider peer to peer file transfer data set in which the individual users, amount of data, and time of transfer have been logged. Visualizing the data transferring between individual users should surface anomalous activity spikes or drops. We can also implement a form of pipeline from clustering to markov chains in order to reveal related values and their order. In this strategy, we are using the clusters as markov states. In addition to these two methods, we can aso use __information theory__ as anomalies contain abnormally high information and the data points which present the highest potential irregularity are likely to prove to be anomalous. 

More traditional techniques can be employed in the case of unordered multivariate collective anomalies. Consider an example of a weeks worth of temperature measurements from random cities. In this case, the simplest method is likely to be clustering. The output of this should result in anomalies falling within smaller clusters than the others, or outside of clusters altogether. Nearest neighbor based algorithms are also useful and for the same reasons as clustering. Anomalies will often appear near to other anomalies and thus be easier to detect with a labeled training set, which is assuming that a non-labeled data set was accurately labeled by clustering.
