# Time Series

## Data

Time series analysis is particular susceptible to corrupted data. Errors or missing values create a greater issue for forecasting due to the often smaller sample sizes and the need for consistency. Data used for forecasting, like all other forms, can suffer from inappropriate _granularity_, which refers to the data coverage. Data covering geographical area, operational times, or population can provide inappropriate levels of granularity so it is important to always consider what is necessary for modeling. Data reported at unnecessarily granular levels will be more difficult and intensive to process. This is true for all statistical modeling, but forecasting requires an additional consideration of _temporal frequency_ which refers to the frequency of the observations. If the observations are taken each second but the forecast requires weekly predictions, the data should be aggregated as is appropriate for the forecast.

## Time Series Components

Time series can be decomposed into three basic components: _level_, _trend_, and _seasonality_. The level is the average value of the series. The trend is the difference in values between the periods of the observations. The seasonality is the cyclical, short-term behavior observable within a given series. Any non-systematic part of the data is the _noise_. 

The evaluation of the time series components are considered to be either _additive_, $y_t = level + trend + seasonality + noise$,  or _multiplicative_, $y_t = level * trend * seasonality * noise$.

## Data Pre-Processing

Common forecasting methods such as ARIMA and smoothing methods require consistent data as they model the relationships between consecutive observational periods, thus missing data precludes applying these methods. Some issues, such as unevenly spaced series by time interval, can be resolved by methods similar to those used for missing data in other contexts. 


# Evaluating Model Performance

## Data Partitioning

With all modeling data, we need to divide the data into a training set and a validation set. In the context of forecasting, this division requires an additional consideration of dividing the data consecutively such that the validation set is a temporally contiguous data set in contrast with other modeling contexts where the validation set is typically a random selection. In order to effectively validate a model of temporal relationships, it is necessary to have this contiguous validation set which is sufficiently large to cover any seasonality present in the data in order to properly test this model. Typically, after the model has been trained and validated, the data is merged such that the model can be trained on the entire data set to take advantage of all available data.

## Forecasts Benchmarks and Accuracy

A common baseline metric for forecast evaluation is what's known as a _naive forecast_, a very simple model that returns the value of the prior observation as any given prediction. In other words, the predicted value for tomorrow is the known value of today, or $F_{t+k} = y_t$. In the classification context, a good prediction benchmark is a simple majority class prediction, which is to say the prediction is the class most present in the time series.

Model evaluation for forecasts can be as simple as the evaluation for any other model. Measures such as MAE and RMSE can be relied upon to reflect the quality of the model against the selected validation data set.

## Evaluating Forecast Uncertainty

Unlike something like a regression a model, forecasts are validated to evaluate the forecast uncertainty. The range of a forecast is referred to as a _prediction interval_, each of which has it's own level of uncertainty. The reason this uncertainty is important is due to the fact that forecast errors are not normally distributed. For example, a forecast of $f_t = 25000$ is not as informative as a 90% certainty that the value will fall within the range of 15000 and 27000, or a 90% interval of [15000, 27000]. It should be noted that the prediction of $f_t = 25000$ is not in the middle of this range and the prediction is thus not normally distributed. Forecast intervals are _prediction cones_, named so for their shapes moving forward in the series. This cone shape is indicative of the ever growing uncertainty moving forward from the last known observation, with the likely range of the predictions increasing.

_Roll-forward validation_ is a validation method used for forecasts which sequentially moves forward through a series decreasing the validation period with each step of the algorithm and averaging the error rates when it is done. This method maximizes the available data by moving progressively from a small training set and a large validation set to a large training set and a small validation set in a manner of simulated deployment of the model. This can be thought of as a time series adapted form of leave-one-out cross validation.


# Forecasting Methods

## Extrapolation Methods, Econometric Models, and External Information

Forecasting methods fall into one of three categories: _extrapolation_, _econometric models_, and _external information_. _Extrapolation methods_ are likely the best known and most traditional being a forecast based upon the known historical data within a time series. _Econometric models_ attempt to cross-correlate multiple time series to establish causation, often involving engaging time series as inputs into other series. _External information_ is a method wherein alternative sources, sometimes seemingly unrelated, are used as indicators for the series. These are most often identified as heuristics such as the "leading lipstick indicator" which predicts economic downturns based upon increased sales of cosmetics. Another is the close correlation between ice cream consumption and shark attacks, both of which are actually due to a rise in temperature.

# Smoothing Methods

## Moving Averages

_Moving averages_ are a set of smoothing algorithms which simply present a mean value for a consecutive subset of observations which are quite useful for characterizing trends but perform very poorly when used for forecasting purposes. There are two kinds of moving averages: _centered moving average_ and _trailing moving average_. The difference between these two comes down to the placement of the consecutive subset of observations. These observations are represented by $w$, an integer value representing the number of observations included in the average. The centered moving average unsuprisingly centers the subset around the given observation and is defined by

$$CMA_t = (y_{t - (w - 1)/2} + ... + y_{t-1} + y_t + y_{t + 1} + ... + y_{t + (w - 1)/2}) / w$$

The trailing moving average begins with the given observation and then looks backwards into the series and is defined by

$$TMA_t = (y_t + y_{t - 1} + ... + y_{t - w + 1}) / w$$

```{r}
library(zoo)
# centered moving average
ma.centered = ma(ridership.ts, order=12)
# trailing moving average
ma.trailing = rollmean(ridership.ts, k=12, align="right")
```

## Differencing

Moving averages are good for visualizing trends, but insufficient for forecasting. In order to effectively forecast, the trend or seasonality needs to be removed from a series. A simple way to do so is _differencing_. This what can be referred to as a _lag-k_ operation where $k$ represents an observational index, which is to say $y_{t - k}$ is subtracted from $y_t$. For instance, _lag-7_ on a daily series subtracts last week's value for a given day from each day's value, or _lag-1_ on the same daily series would subtract the prior day's value from the given day. The best approach for choosing $k$ is basing this upon the seasonal pattern to be removed. $k$ should be equal to the seasonal pattern with $M$ seasons. Differencing can be applied twice in order to remove both trend and seasonality.

## Exponential Smoothing

Exponential smoothing is a popular option for forecasting. In contrast with moving averages which only considers $w$ observations, exponential smoothing calculates a _weighted average_ of all historical values such that observations decrease in influence as they get older. This is calculated by

$$F_{t + 1} = \alpha y_t + \alpha (1 - \alpha) y_{t - 1} + \alpha (1 - \alpha)^2 y_{t - 2} + ... $$

where $\alpha$ is a smoothing constant chosen by the user. $\alpha$ values closer to $1$ indicate faster learning and aggressive smoothing and values closer to $0$ slower learning and less smooth prediction lines. Commonly effective values for $\alpha$ fall between $0.1$ and $0.2$ but this ought to be chosen by validation.

While simple and easily explained, exponential smoothing should only be applied in cases of series with no trend or seasonality. When dealing with series with trend or seasonality, one can remove these components with differencing prior to applying exponential smoothing. An alternative method is the use of a more sophisticated form of exponential smoothing. _Double exponential smoothing_, otherwise known as _Holt's linear trend model_, can be used on series with a trend. This model comes in two flavors: _additive_ and _multiplicative_. The additive version of this model, the forecast for at $k$ is given by

$$F_{t + k} = L_t + kT_t$$

where

$$L_t = \alpha y_t + (1 - \alpha)(L_{t-1} + T_{t-1})$$

$$T_t = \beta (L_t - L_{t-1}) + (1 - \beta) T_{t-1}$$

As with simple exponential smoothing, $\alpha$ and $\beta$ are both smoothing constants chosen by the user falling between $0$ and $1$ with higher values providing more aggressive smoothing and lower values providing less aggressive smoothing. Another way of looking at these smoothing constants is that they determine the priority to be placed on more recent information, higher values granting more recent information greater influence.

The multiplicative model is forecasted for $k$ by

$$F_{t+k} = L_t * T^k_t$$

where 

$$L_t = \alpha y_t + (1 - \alpha) (L_{t-1} * T_{t -1 })$$

$$T_t = \beta (L_t / L_{t-1}) + (1 - \beta) T_{t-1}$$

As in the model, there are two kinds of errors in exponential smoothing: _additive_ and _multiplicative_. Additive errors assume that levels change by fixed amounts from one period to another and is represented as

$$y_{t + 1} = L_t + T_t + e_t$$

Multiplicative errors assume that these levels change by factors and is represented as

$$y_{t + 1} = (L_t + T_t) * (1 + \epsilon_t)$$

It is worth noting that in both moving averages and exponential smoothing, the user ultimately determines the priority to be placed on newer data over older data. This is done in exponential smoothing inherently, but the choice of $w$ in a moving average insinutates this priority as only $w$ observations will be considered for the average and everything else will be ignored.

When a series contains both seasonality and trend, the double exponential smoothing model can be extended to account for the season at observation $t + k$. This method is known as the _Holt-Winter's exponential smoothing_ method. Within this context, the seasonality can be multiplicative by seasonal differences adopt percentage values or additive where the seasonal differences are fixed values. These character differences within the series can be accounted for intuitively. For instance, a multiplicative seasonality coupled with an additive trend is modeled by

$$F_{t+k} = (L_t + kT_t) s{t + k - M}$$

for $M$ seasons where

$$L_t = \alpha y_t / S_{t - M} + (1 - \alpha) (L_{t-1} + T_{t-1})$$

$$T_t = \beta (L_t - L_{t-1}) + (1 - \beta) T_{t-1}$$

$$S_t = \gamma (y_t / L_t) + (1 - \gamma) S_{t-M}$$

An additive seasonality and additive trend is modeled by

$$F_{t + k} = L_t + kT_t + S_{t + k - M}$$

where

$$L_t = \alpha (y_t - S_{t - M}) + (1 - \alpha)(L_{t -1 } + T_{t - 1})$$

$$T_t = \beta (L_t - L_{t - 1}) + (t - \beta) T_{t - 1}$$

$$S_t = \gamma (y_t - L_t) + (1 - \gamma) S_{t - M}$$

and so on.

## Extensions of Exponential Smoothing

While the Holt-Winter's expnential smoothing method is useful, series with multiple seasonal cycles are quite common. For example, most retail stores will see sales on a given day rise in the early evening compared to the rest of the day, and on weekends compared to the rest of the week. An hourly log of store sales will then present two seasonal cycles within this series. In order to account for this, a double-seasonal model will be needed. This can be accomplished by further extending the Holt-Winter's method using four updating equations for each series component: level, trend, cycle 1, and cycle 2. This is given by

$$F_{t + k} = (L_t + kT_t) * S^{(1)}_{t + k - M_1} * S^{(2)}_{t + k - M_2}$$

where

$$L_t = \alpha Y_t / (S^{(1)}_{t-M_1} S^{(2)}_{t-M_2}) + (1 - \alpha)(L_{t-1} + T_{t - 1})$$

$$T_t = \beta (L_t - L_{t - 1}) + (1 - \beta) T_{t - 1}$$

$$S^{(1)}_t = \gamma_1 (y_t / (L_t S^{(2)}_{t - M_2})) + (1 - \gamma_1) S^{(1)}_{t - M_1}$$

$$S^{(2)}_t = \gamma_2 (y_t / (L_t S^{(1)}_{t - M_1})) + (1 - \gamma_2) S^{(2)}_{t - M_2}$$

# Regression-based Models: Capturing Trend and Seasonality

Linear methods in the context of time series are no different from linear methods in other areas of statistical learning. Seasonality can be represented in a linear model of a time series by a categorical identifier such as year, quarter, month, week, or day. The choice of granularity largely depends upon the seasons present in the time series. The methods for accounting for trends and seasonality in a linear model of a time series can be combined to accomodate both simultaneously within the same model.

# Regression-based Models: Autocorrelation & External Information

Regression methods can be very useful in modeling time series. Specifically, they can be used to develop an _autoregressive_ model which quantifies the correlation between consecutive observations within a time series. This correlation is known as _autocorrelation_.

## Autocorelation

While regression methods are very useful in modeling time series, ordinary implementations do not accurately model the correlation between chronological observations known as autocorrelation. This is important as consecutive periods within time series are often correlated. The autocorrelation can be calculated by evaluating the correlation between the series and the lagged values of the series. The autocorrelations are named for the lag correlated. For instnace, a lag-1 autocorrelation is the correlation between the series and the lag-1 values, whereas the lag-2 autocorrelation is the correlation between the series and the lag-2 values, etc. The autocorrelation of a series can be very informative as they can be indicative of seasonality within the series. 

## Improving Forecasts by Capturing Autocorrelation: AR and ARIMA Models

_AutoRegressive Integrated Moving Average_, or _ARIMA_, models are a class of autoregressive methods very much like linear regression that models autocorrelation and forecast errors within a series. The key distinction between linear regression and ARIMA models is that ARIMA models specifically target time series and thus use prior observed values in a series as predictors for future values. AutoRegressive models are developed by level of _order_ of differencing. Order of 0 indicates no differencing, order of 1 removes linear trends by differencing the series once, order of 2 removes quadratic trends by differencing the series twice, and so on. For instance, an AR(2) model, or an autoregressive model of order 2, is defined as $y_t = \beta_0 + \beta_1 y_{t - 1} + \beta_2 y_{t - 2} + \epsilon_t$. This is effectively the same as producing a linear regression model which uses the lagged observations as predictors and outputs the series.

These ARIMA models can be utilized in two ways: deploying an ARIMA model and thus inserting found autocorrelation into a regression model, or producing a _second-level forecast_ on the residual series. These second-level forecasting models for residuals can be produced by following these steps:

1. Forecast the series to $k$ steps by any forecasting method to calculate $F_{t + k}$
2. Model the error of the forecast to $k$ steps by an AR model to calculate $\epsilon_{t + k}$
3. Use the errors generated by the AR model to improve the original forecast model to calculate $F^*_{t + k} = F_{t + k} + \epsilon_{t + k}$

The ARIMA model is characterized by the parameters $(p, d, q)$. These parameters dictate the primary features of the model to be trained. $p$ is the number of AR terms, $d$ is the number of times the series is differenced prior to the application of an ARIMA model, and $q$ is the number of moving average terms. It can be useful in the context of short term forcasting to add yet another AR layer given an $k$ order AR model can only effectively forecast the next $k$ periods, after which the forecast is relying upon previously forecasted, or estimated, data. While using derived data can sometimes be useful for statistical learning models, it does not always create the most reliable prediction and is a less than ideal solution.

AR and ARMA models can only be used on series without trend or seasonality. 

AR(p) models are defined by

$$y_t = \beta_0 + \beta_1 y_{t-1} + \beta_2 y_{t - 2} + ... + \beta_p y_{t - p} + \epsilon_t$$

ARMA(p, q) models are defined by

$$y_t = \beta_0 + \beta_1 y_{t-1} + \beta_2 y_{t - 2} + ... + \beta_p y_{t - p} + \epsilon_t + \theta_1 \epsilon_{t - 1} + \theta_2 \epsilon_{t - 2} + ... + \theta_q \epsilon_{t - q}$$

## Evaluating Predictability

Another useful method of evaluating series is determining whether or not the series is what is known as a _random walk_. A series that is a random walk presents itself as such by varying consecutive values randomly and forecasts from these are functionally equivalent to the naive forecast. Evaluating whether or not a series is a random walk can be accomplished by fitting an AR(1) model to the series. If the slope coefficient for the model is equal to 1 the series is a random walk, otherwise it is not.

## Including External Information

In addition to relationships between consecutive periodic observations, regression models can also make evident patterns within a given series that do not directly relate to the features present within the dataset. For instance, such influential factors as _special events_, _interventions_, _policy changes_, and other more generalized _correlated external series_ often play a role in real world outcomes but may not be directly represented in the data. _Special events_ can be Black Friday for a retail store or extreme sales in a service department and such events can significantly skew a series forecast if not properly accounted for. _Interventions_ are similar to special events but would be more likely to be entirely outside the control of the operator in a given situation such as inclement weather causing an evacuation of a service area or school holidays impacting daycare attendance. _Policy changes_ are typically impacts to a series by external but directly influential factors such as legislation or policy action by influential industries. _Correlated external series_ can be related data, but not always (ie, aforementioned cosmetic sales correlating to broader economic conditions or ice cream consumption to shark attacks). These can be more difficult to find and identify but can be very helpful as there exists the virtual certainty that for any given datapoint, there exists some seemingly unrelated factor that influences or is influenced by the modeled series. When these can be identified, they are integrated in a two step process:

1. Remove the seasonality and trend from both series if present
2. Fit the regression model $Y^*_t$ using the lag values of $y^*_t$ or $x^*_t$ as predictors

# Forecasting Binary Outcomes

The goal of binary forecast is not always as simple as will an event occur or not. This same method can be used to forecast the vector of change (ie, will a stock price move up or down?) or the amplitude of a change (ie, will the temperature rise above 100 degrees?). Just as with regression forecasts, the components of the series need to be modeled by seasonality, trend, and autocorrelation. 

