# 8 Tree-Based Methods

Tree-based methods involve _stratifying_ or _segmenting_ the predictor into a number of simple regions. In order to make a prediction, we typically use the mean or the mode of the training observations in the region to which it belongs. _Decision tree_ methods are simple and useful for interpretation, but are typically not competitive with the best supervised learning methods in terms of prediction accuracy.

_Bagging_, _random forests_ and _boosting_ involve producing multiple trees which are then combined to yield a single consensus prediction. Combining a large number of trees can often result in dramatic improvements in prediction accuracy at the expense of interpretability.

___

## The Basics of Deicision Trees

### Regression Trees

In keeping with the tree analogy, the subdivided regions of a decision tree are known as _terminal nodes_ or _leaves_ of the tree. The points along the tree where the predictor space is split are referred to as _internal nodes_. The segments of the trees that connect the nodes are referred to as _branches_.

#### Prediction via Stratification of the Feature Space

There are roughly 2 steps to building a regression tree:

1. We divide the predictor space - that is, the set of possible values for $X_1, X_2, ..., X_p$ - into $J$ distinct and non-overlapping regions, $R_1, R_2, ..., R_J$.
2. For every observation that falls into the region $R_j$, we make the same prediction, which is simply the mean of the response values for the training observations in $R_j$.

In theory, the regions $J$ could be divided in any shape. However, we choose to divide the predictor space into high-dimensional rectangles, or _boxes_, for simplicity and for ease of interpretation of the resulting predictive model. The goal is to find boxes $R_1, ..., R_j$ that minimize $RSS$ as defined by

$\sum^J_{j=1} \sum_{i \in R_j} (y_i - \hat{y}R_j)^2$

where $\hat{y}R_j$ is the mean response for the training observations within the $j^{th}$ box. It is computationally infeasible to consider every possible partition of the feature space into $J$ boxes so we take a _top-down, greedy_ approach that is known as _recursive binary splitting_.

The approach is _top-down_ because it begins at the top of the tree (at which point all observations belong to a single region) and then successively splits the predictor space; each split is indicated via two new branches further down on the tree. It is _greedy_ because at each step of the tree-building process, the _best_ split is made at that particular step, rather than looking ahead and picking a split that will lead to a better tree in some future step.

In order to perform recursive binary splitting, we consider predictors $X_1, ..., X_p$, and all possible values of the cutpoint's for each of the predictors, and then choose the predictor and cut point such that the resulting tree has the lowest $RSS$.

#### Tree Pruning

Building a tree without due consideration can lead to overly complicated, overfitted trees. A good way to avoid this is to grow a very large tree $T_0$ and then _prune_ it back in order to obtain a _subtree_. Given a subtree, we can estimate its test error using cross-validation or the validation set approach, but we need a way to select a small of candidate subtrees for consideration.

_Cost complexity pruning_ - also known as _weakest link pruning_ - gives us a way to do just this. Rather than considering ever possible subtree, we consider a sequence of trees indexed by a nonnegative tuning parameter $a$. For each value of a there corresponds a subtree $T \subset T_0$ such that

$\sum^{|T|}_{m=1} \sum_{i:y_i \in R_m}(y_i - \hat{y}_{R_m})^2 + a|T|$

is as small as possible. Here $|T|$ indicates the number of terminal nodes of the tree $T$, $R_m$ is the rectangle (ie, the subset of predictor space) corresponding to the $m^{th}$ terminal node, and $\hat{y}_{R_m}$ is the predicted response associated with $R_m$ - that is, the mean of the training observations in $R_m$. The tuning parameter a controls a trade off between the subtrees complexity and its fit to the training data. When $a=0$, then the subtree $T$ will simply equal $T_0$, because then the above formula is just measuring the training error. However, as $a$ increases, there is a price to pay for having a tree with many terminal nodes, and so the quantity will tend to be minimized for a smaller subtree.

#### Algorithm for Building a Regression Tree

1. Use recursive binary splitting to grow a large tree on the training data, stopping only when each terminal node has fewer than some minimum number of observations.
2. Apply cost complexity pruning to the large tree in order to obtain a sequence of best subtrees, as a function of $a$.
3. Use k-fold cross-validation to choose $a$. That is, divide the training observations in $k$ folds. For each $k=1, ..., K$,
    a. Repeat steps 1 & 2 on all but the $k^{th}$ fold of the training data
    b. Evaluate the mean squared prediction error on the data in teh left-over $k^{th}$ fold, as a dunction of $a$
      Average the results for each value of $a$, and pick $a$ to minimize the average error.
4. Return the subtree from step 2 that corresponds to the chosen value of $a$

### Classification Trees

A classification tree is similar to a regression tree, however it predicts qualitative responses rather than quantitative responses. For a classification tree, we predict that each observation belongs to the _most commonly occurring class_ of training observations in the region to which it belongs. In interpreting the results of a classification tree, we are often interested not only in the class prediction corresponding to a particular terminal node region, but also in the _class proportions_ among the training observations that fall into that region.

In the classification context, $RSS$ cannot be used as a criterion for making the binary splits. A natural alternative to $RSS$ is the _classification error rate_. Since we plan to assign an observation in a given region to the _most commonly occurring class_ of trining observations in that region, the classification error rate is simply the fraction of the training observations in that region that do not belong to the most common class:

$E = 1 - max_k (\hat{p}_{mk})$

However, it turns out that classification error is not sufficiently sensitive for tree-growing, so 2 alternatives are used in practice:

1. _Gini index_
    This is a measure of total variance across the $K$ classes and is measured by

    $G = \sum^K_{k=1}\hat{p}_{mk}(1 - \hat{p}_{mk})$
2. _Cross-entropy_
    Conceptually similar to Gini index, but measured by

    $D = -\sum^K_{k=1} \hat{p}_{mk} log \hat{p}_{mk}$

    Like the Gini index, the cross-entropy will take on a small value if the $m^{th}$ node is pure.

    When building a classification tree, either the Gini index or the cross-entropy are typically used to evaluate the quality split, since these two approaches are more sensitive to node purity than is the classification error rate.

### Tree vs Linear Models

Linear regression assumes a model of the form

$f(X) = \beta_0 + \sum^P_{j=1} x_j\beta_j$

whereas regression trees assume a model of the form

$f(X) = \sum^M_{m=1} c_m * 1_{(x \in R_m)}$

where $R_1, ..., R_m$ represent a partition of feature space. Which model is better depends upon the problem your trying to solve. Another consideration is that interpretability of one model or the other may be better in varying circumstances.

### Advantages and Disadvantages of Trees

1. Trees are easier to explain to people 
2. Some believe that decision trees more closely mirror human decision making
3. Trees can be displayed and easily understood graphically
4. Trees can easily handle qualitative predictors
5. Trees generally do not have the same level of predictive accuracy as some of the other regression and classification approaches
6. Trees can be very non-robust; i.e., a small change in data can lead to large changes in the final estimated tree

The disadvantages can often be overcome by aggregated tree methods such as _bagging_, _random forests_, and _boosting_.

___

## Bagging, Random Forests, Boosting

### Bagging

Decision trees suffer from _high variance_. A procedure with _low variance_ will yield similar results if applied repeatedly to distinct data sets; linear regression tends to have low variance, if the ratio of $n$ to $p$ is moderately large. _Bootstrap aggregation_, or _bagging_, is a general purpose procedure for reducing the variance of a statistical learning method.

_Averaging a set of observations reduces variance._ Hence a natural way to reduce the variance and increase the prediction accuracy of a statistical learning method is to take many training sets from the population, build a separate prediction model using each training set, and average the resulting predictions. In other words, we could calculate $f^1(x), f^2(x), ..., f^B(x)$ using $B$ separate training sets, and average them in order to obtain a single low-variance statistical learning model, give by

$\hat{f}_{avg}(x) = \frac{1}{B} \sum^B_{b=1}\hat{f}^b(x)$

Of course, this is not practical because we generally do not have access to multiple training sets. Instead, we can _bootstrap_, by taking repeated samples from the (single) training data set. In this approach we generate $B$ different bootstrapped training data sets. We then train our method on the $b^{th}$ bootstrapped training set in order to get $f^{tb}(x)$, and finally average all the predictions, to obtain

$\hat{f}_{bag}(x) = \frac{1}{B} \sum^B_{b=1} \hat{f}^{tb}(x)$

This is called _bagging_.

To apply bagging to regression trees, we simply construct $B$ regression trees using $B$ bootstrapped training sets, and average predictions. These trees are grown deep, and are not pruned. Hence each individual tree has high variance, but low bias. Averaging these $B$ trees reduces the variance.

### Out-of-Bag Error Estimation

There is a straight forward way to estimate error with a bagged model. Recall that the key to bagging is that trees are repeatedly fit to bootstrapped subsets of the observations. One can show that on average, each bagged tree makes use of around two-thirds of the observations. The remaining one-third of the observations not used to fit a given bagged tree are referred to as the _out-of-bag_ (OOB) observations. It can be shown that with $B$ sufficiently large, OOB error is virtually equivalent to leave-one-out cross-validation error.

### Variable Importance Measures

Bagging improves prediction accuracy at the expense of interpretability. One can obtain an overall summary of the importance of each predictor using the $RSS$ (for bagging regression trees) or the Gini index (for bagging classification trees).

___

## Random Forests

_Random forests_ provide an improvement over bagged trees by way of a small tweak that _decorrelates_ the trees. As in bagging, we build a number of decision trees on bootstrapped training samples. But when building these decision trees, each time a split in a tree is considered, _a random sample of $m$ predictors_ is chosen as split candidates from the full set of $p$ predictors. The split is allowed to use only one of those $m$ predictors. A fresh sample of $m$ predictors is taken at each split and typically we choose $m \approx \sqrt{p}$ - that is, the number of predictors considered at each split is approximately equal to the square root of the total number of predictors. More specifically, random forests _decorrelate_ the trees because, on average, $(p * m)/p$ of the splits will not even consider a given strong predictor.

Using a small value of $m$ in building a random forest will typically be helpful when we have a large number of correlated predictors.

___

## Boosting

Like bagging, _boosting_ is a general approach that can be applied to many statistical learning methods for regressino or classification. The methodology is very similar excepting that in boosting the trees are grown _sequentially_: each tree is grown using information from previously grown trees. Boosting deos not involve bootstrap sampleing; instead each tree is fit on a modified version of the original data set.

The boosting algorithm:

1. Set $f(x)=0$ and $r_i = y_i$ for all $i$ in the training set
2. Set $b = 1, 2, ..., B_n$ repeat
    a. Fit a tree $f^b$ with $d$ splits ($d+1$ terminal nodes) to the training data ($x,r$)
    b. Update $f$ by adding in a shrunken version of the new tree: $f(x) \leftarrow f(x) + \lambda f^b(x)$
    c. UPdate the residuals: $r_i \leftarrow r_i - \lambda f^b(x_i)$
3. Output the boosted mdoel
    $f(x) = \sum^B_{b=1} \lambda \hat{f}^b(x)$

Unlike fitting a single large decision tree to the data, which amounts to _fitting the data hard_ and potentially overfitting, the boosting approach instead _learns slowly_. Given the current model, we fit a decision tree to the residuals from the model. That is, we fit a tree using the current residuals, rather than the outcome $Y$, as the response. We then add thsi new decision tree into the fitted function in order to update the residuals. Each of these trees can be rather small, with just a few terminal nodes, determined by the parameter $d$ in the algorithm. By fitting small trees to the residuals, we slowly improve $\hat{f}$ in areas where it does not perform well. The shrinkage parameter $\lambda$ slows the process down even further, allowing more and different forms of trees to attach the residuals. In general, statistical learning approaches that _learn slowly_ tend to perform well. Note that in boosting, unlike in bagging, the construction of each tree depends strongly on the trees that have already been grown.

Boosting has 3 tuning parameters:

1. The numebr of trees $B$: boosting can overfit if $B$ is too large; use cross-validation to select $B$
2. The shrinkage parameter $\lambda$: controls rate at which boosting learns; typical values are 0.01 or 0.001
3. The number $d$ of splits in each tree: controls complexity of the boosted ensemble; often $d=1$ works well. More generally $d$ is the interaction depth, and controls the interaction order of the boosted model, since $d$ splits can involve at most $d$ variables.

One of the differences between random forests and boosting is that because the growth of a particular tree takes into account the other trees that have already been grown, smaller trees are typically sufficient in boosting.