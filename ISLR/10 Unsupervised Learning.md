# 10 Unsupervised Learning

_Unsupervised learning_ is a set of statistical tools intended for the setting in which we have only a set of features $X_1, X_2, ..., X_p$ measured on $n$ observations. We are not interested in preodiction, because we do not have an associated response variable $Y$. Rather, the goal is to discover interesting things about the measurements on $X_1, X_2, ..., X_p$. We will explore two particular types of unsupervised learning:

1. _Principal Components Analysis_: a tool for data visualization or data pre-processing before supervised techniques are applied, and
2. _Clustering_: a broad class of methods for discovering unknown subgroups in data.

___

## The Challenge of Unsupervised Learning
Unsupervised learning is often performed as part of an _exploratory data analysis_.

___

## Principal Component Analysis
When faced with a large set of correlated variables, principal components allow us to summarize this set with a smaller number of representative variables that collectively explain most of the variability in the original set. To perform principal components regression, we simply use principal components as predictors in a regression model in place of the original larger set of variables.

_Principal component analysis_ (PCA) refers to the process by which principal components are computed, and the subsequent use of these components in understanding the data.

### What Are Principal Components?
When wanting to visualize a data set where $p$ is large, we would like to find a low-dimensional representation of the data that captures as much of the information as possible. PCA provides a tool to do just this. It finds a low-dimensional representation of a data set that contains as much as possible of the variation. PCA seeks a small number of dimensions that are as interesting as possible, where the concept of _interesting_ is measured by the amount that the observations vary along each dimension, otherwise known as _principal components_.

The _first principal component_ of a set of features $X_1, X_2, ..., X_p$ is the normalized linear combination of the features

$Z_1=\phi_{11}X_1+\phi_{21}X_2+...+\phi_{p1}X_p$

that has the largest variance. The first principal component loading vector solves the optimization problem

$max_{\phi_{11}, ..., \phi_{p1}}\{\frac{1}{n}\sum^n_{i=1}(\sum^p_{j=1}\phi_{j1}x_{ij})\}\ subject\ to\ \sum^p_{j=1}\phi^2_{j1}=1$

The objective in the above can be considered to be $\frac{1}{n}\sum^n_{i=1}z^2_{i1}$. We refer to $x_{11}, ..., z_{n1}$ as the _scores_ of the first principal component.

### Another Interpretation of Principal Components
We describe the principal component loading vectors as the directions in feature space along which the data vary the most, and the principal component scores as projections along these directions. However, an alternative interpretation for principal components can also be useful: principal components provide low-dimensional linear surfaces that are _closest_ to the observations. 

The first principal component loading vector has a very special property: it is the line in _p_-dimensional space that is _closest_ to the _n_ observatinos (using average squared Euclidean distance as a measure of closeness). The first two principal components of a data set span the plane that is closest to the _n_ observations, in terms of average squared Euclidean distance. The first three principal components of a data set span the three-dimensional hyperplane that is closest to the _n_ observations, and so forth.

### More on PCA
_The results obtained when we perform PCA will also depend on whether the variables have been individually scaled._ In linear regression, multiplying a variable by a factor of $c$ will simply lead to multiplication of the corresponding coefficient estimate by a factor of $I/c$, and thus will have no substantive effect on the model obtained.

Considering the example crime data, if we perform PCA on the unscaled variables, then the first principal component loading vector will have a very large loading for **Assault**, since that variable has by far the higest variance. Comparing this to the left-hand plot in the text, we see that scaling does indeed have a substantial effect on the results obtained. _Because it is undesirable for the principal components obtained to depend on an arbitrary choice of scaling, we typically scale each variable to have standard deviation one before we perform PCA._

#### The Proportion of Variance Explained
We have seen on the **USArrests** data set that we can summarize the 50 observations and 4 variables using just the first two principal component score vectors and the first two principal component loading vectors. We can now ask a natural question: how much of the information in a given data set is lost by projecting the observations onto the first few principal components? That is, how much of the variance in the data is _not_ contained in the first few principal components? More generally, we are interested in knowing the _proportion of vairance explained_ (PVE) by each principal component. The _total variance_ present in a data set (assuming that the variables have been centered to have mean zero) is defined as 

$\sum^p_{j=1}Var(X_j)=\sum^p_{j=1}\frac{1}{n}\sum^n_{i=1}x^2_{ij}$

and the variance explained by the $m^{th}$ principal component is

$\frac{1}{n}\sum^n_{i=1}z^2_{im}=\frac{1}{n}\sum^n_{i=1}(\sum^p_{j=1}\phi_{jm}x_{ij})^2$

Therefore, the PVE of the $m^{th}$ principal component is given by 

$\frac{\sum^n_{i=1}(\sum^p_{j=1}\phi_{jm}x_{ij})^2}{\sum^p_{j=1}\sum^n_{i=1}x^2_{ij}}$

The PVE of each principal component is a positive quantity.

#### Deciding How Many Principal Components to Use
In general, a $n x p$ data matrix **X** has $min(n-1, p)$ distinct principal components. We would like to use the smallest number of principal components required to get a _good_ understanding of the data. We typically decide on the number of principal components required to visualize the data by examining a _scree plot_. This type of visual analysis is inherently _ad hoc_. Unfortunately, there is no well-accepted objective way to decide how many principal components are _enough_.

### Other Uses for Principal Components
Many statistical techniques, such as regression, classification, and clustering, can be easily adapted to use the $n x M$ matrix whose columns are the first $M \ll p$ principal component score vectors, rather than using the full $n x p$ data matrix. This can lead to _less noisy_ results, since it is often the case that the signal (as opposed to the noise) in a data set is concentrated in its first few principal components. 

___

## Clustering Methods

_Clustering_ refers to a very broad set of techniques for finding _subgroups_, or _clusters_, in a data set. Both clustering and PCA seek to simplify the data via a small numbers of summaries, but their mechanisms are different:

- PCA looks to find a low-dimensional representation of the observations that explain a good fraction of the variance;
- Clustering looks to find homogeneous subgroups among the observations.

The task of performing market segmentation is an example of clustering that amounts to clustering the people in a the data set by their attributes.

In _$K$-means clustering_, we seek to partition the observations into a pre-specified number of clusters. In _hierarchical clustering_, we do not know in advance how many clusters we want; in fact, we end up with a tree-like visual representation of the observations, called a _dendrogram_, that allows us to view at once the clusterings obtained for each possible number of clusters, from 1 to $n$.

### $K$-Means Clustering
$K$-means clustering is a simple and elegant approach for partitioning a data set into $K$ distinct, non-overlapping clusters. Let $C_1, ..., C_K$ denote sets containing the indices of the observations in each cluster. These sets satisy two properties:

1. $C_1 \cup C_2 \cup ... \cup C_K=\{1, ..., n\}$. In other words, each observation belongs to at least one of the $K$ clusters.
2. $C_k \cap C_{k'}=\emptyset$ for all $k\neq k'$. In other words, the clusters are not overlapping: no observation belongs to more than one cluster.

The idea behind $K$-means clustering is that a _good_ clustering is one for which the _within-cluster variation_ is as small as possible. This is represented as

$min_{C_1, ..., C_K}\{\sum^K_{k=1}W(C_k)\}$

This formula says that we want to partition the observations into $K$ clusters such that that total within-cluster variation, summed over all $K$ clusters, is as small as possible.

In order to do this, we need to define the within-cluster variation which is most commonly done by use of _squared Euclidean distance_, which is defined as

$W(C_k)=\frac{1}{|C_k|}\sum_{i,i' \in C_k}\sum^p_{j=1}(x_{ij}-x_{i'j})^2$

where $|C_k|$ denotes the number of observations in the $k^{th}$ cluster. In other words, the within-cluster variation for the $k^{th}$ cluster is the sum of all of the pairwise squared Euclidean distances between the observations in the $k^{th}$ cluster, divided by the total number of observations in the $k^{th}$ cluster.

A very simple algorithm can be shown to provide a local optimum - a _pretty good solution_ - to the $K$-means optimzation problem.

#### $K$-Means Clustering Algorithm
1. Randomly assign a number, from 1 to $K$, to each of the observations. These serve as initial cluster assignments for the observations
2. Iterate until the cluster assignments stop changing:
    a. For each of the $K$ clusters, compute the cluster _centroid_. The cluster centroid is the vector of the $p$ feature means for the observation in the $k^{th}$ cluster.
    b. Assign each observation to the cluster whose centroid is closest (where _closest_ is defined using Euclidean distance).

$K$-means clustering derives its name from the fact that in step 2(a), the cluster centroids are computed as the mean of the observations assigned to each cluster.

### Hierarchical Clustering

One potential disadvantage of $K$-means clustering is that it requires us to pre-specify the number of clusters $K$. _Hierarchical clustering_ is an alternative approach which does not require that we commit to a particular choice of $K$. _Bottom-up_ or _agglomerative_ clustering is the most common type of hierarchical clustering, and refers to the fact that a _dendrogram_ (generally depicted as an upside-down tree).

#### Interpreting a Dendrogram

For any two observations, we can look for the point in the tree where branches containing those two observations are first fused. The height of this fusion, as measured on the vertical axis, indicates how different the two observations are. We cannot draw conclusions about the similarity of two observations based on their proximity along the _horizontal axis_. Rather, we draw conclusions about the similarity of two observations based on the location on the _vertical axis_ where branches containing those two observations first are fused. Hierarchical clustering can sometimes yield _worse_ (ie, less accurate) results than $K$-means clustering for a given number of clusters.

#### The Hierarchical Clustering Algorithm

In order to evaluate a hierarchical cluster, we must define some sort of _dissimilarity_ measure between each pair of observations. Starting out at the bottom of the dendrogram, each of the _n_ observations is treated as its own cluster. The two clusters that are most similar to each other are then _fused_ so that there now are $n-1$ clusters. Next the two clusters that are most similar to each other are fused again, so that there now are $n-2$ clusters.

**Hierarchical Clustering Algorithm**

1. Begin with $n$ observations and a measure (such as Euclidean distance) all the $(^n_2)=n(n-1)/2$ pairwise dissimilarities. Treat each observation as its own cluster.
2. For $i=n, n-1, ..., 2$:
  a. Examine all pairwise inter-cluster dissimilarities among the $i$ clusters and identity the pair of clusters that are least dissimilar (that is, most similar). Fuse these two clusters. The dissimilarity between these two clusters indicates the height in the dendrogram at which the fusion should be placed.
  b. Compute the new pairwise inter-cluster dissimilarities among the $i-1$ remaining clusters.

The concept of dissimilarity between a pair of observations needs to be extended to a pair of _groups of observations_. This extension is achieved by developing the notion of _linkage_, which defines the dissimilarity between two groups of observations. The four most common typws of linkage are _complete_, _average_, _single_, and _centroid_.

#### Choice of Dissimilarity Measure

The choice of dissimilarity measure is very important, as it has a strong effect on the resulting dendrogram. Whether or not it is a good decision to scale the variables before computing dissimilarity measure depends on the application at hand.

### Practical Issues in Clustering

#### Small Decisions with Big Consequences

In order to perform clustering, some decisions must be made.

- Should the observations or features first be standardized in some way? For instance, maybe the variables should be centered to have a mean zero and scaled to have standard deviation one.
- In the case of hierarchical clustering,
  - What dissimilarity measure should be used?
  - What type of linkage should be used?
  - Where should we cut the dendrogram in order to botain clusters?
- In the case of $K$-means clustering, how many clusters should we look for in the data?

Try several different choices, and look for the one with the most useful or interpretable solution.

#### Validating the Clusters Obtained

There exist a number of techniques for assigning a p-value to a cluster in order to assess whether there is more evidence for the cluster than one would expect due to chance. However, there has been no consensus on a single best approach.

#### Other Considerations in Clustering

Since $K$-means and hierarchical clustering force _every_ observations into a cluster, the clusters found may be heavily distorted due to the presence of outliers that do not belong to any cluster. Clustering methods generally are not very robust to perurbations to the data.

#### A Tempered Approach to Interpreting the Results of Clustering

Since clustering can be non-robust, we recommend clustering subsets of the data in order to get a sense of the robustness of the clusters obtained.