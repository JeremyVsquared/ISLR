# Assocation Rules

There are many circumstances in which it can be very useful to examine the relationships between particular items within a dataset. This frequently arises when recommendation engines are needed to identify products often purchsed in coincidence with one being viewed by a user, or which video is related to the currently viewed video in the context of a streaming service. The most basic method of developing such a system is to discover _association rules_ between all the available choices. 

These association rules typically take the form of {flashlight} -> {batteries}, which is to say that when a customer purchases a flashlight they are highly likely based upon transaction history to simultaneously purchase batteries. Such association rules involving itemset subsets will be derived from the larger database of transactions. Association rules are an unsupervised method and as such do not require training.

There is an implementation of the Apriori algorithm, probably the most commonly used association rules algorithm, in the arules package in R.

```r
library(arules)

itemFrequency(transactions, support=0.1)

transactions.rules = apriori(transactions, parameter=list(support=0.006, confidence=0.25, minlen=2))
summary(transactions.rules)
```

# The Apriori Algorithm

The simplest way to find these association rules are to evaluate the presence of every possible subset of the itemsets. When there are large numbers of available items, finding these association rules can be computationally intensive. In a realistic scenario, many of these subsets will rarely, if ever, appear in the true transactional records and thus the computational burden of this process can be greatly alleviated by simply not considering these combinations. The Apriori algorithm is an extremely popular and simple method of reducing the number of subsets that need to be considered.

The basis, and namesake, of the algorithm is the foundational a priori belief known as the _Apriori principle_ that all subsets of a frequently occuring subset must also be frequently occurring. For example, this principle would lead us to believe that the combination of baby food and motor oil can only frequently occur if both baby food and motor oil frequently occur independently. If either of these assertions do not hold true, this combination would not be considered within an association rule search. As another example, consider the presence of the subsets {A, B}, {B, C}, and {A, C}. If it is determined {C} does not frequently occur, then the subsets {A, C}, {B, C}, and {A, B, C} would be excluded from the search since they include {C}.

Many of these determinations are made based upon two metrics: _support_ and _confidence_. _Support_ is the measurement of frequency of a given subset and is calculated by $support(x) = \frac{count(x)}{N}$ where $x$ is a given subset and $N$ is the total number of observations within the dataset. _Confidence_ is the quantification of a given rules accuracy and is calculated by $confidence(x \rightarrow y) = \frac{support(x, y)}{support(x)}$.

Another useful metric of association rules is _lift_, which is a metric that combines support and confidence to evaluate the dependence of one element of the association rule to the other. More practically, in the context of retail data, lift will report the likelihood of purchasing Y given the purchase of X. This is calculated by $lift(X \rightarrow Y) = \frac{confidence(X \rightarrow Y)}{support(Y)}$. A lift of 1 would indicate the probabilities of X and Y are independent, and greater than 1 indicates these probabilities are related.

Discovering valid association rules is done in two steps:

1. Find all subsets above a minimum support threshold
2. Establish rules based upon these subsets which are above a minimum confidence threshold

# Frequent Pattern Growth

The _frequent pattern tree_ algorithm accomplishes a similar outcome as apriori but is more efficient. Where the apriori algorithm can repeatedly scan the dataset, the frequent pattern tree algorithm only scans the dataset twice. In the first pass, the algorithm counts each item. The second pass builds the tree, discarding items if they don't meet a specified threshold, which is used to develop the data structure storing frequent itemsets. 

This process can be recursed and the trees trimmed in the _FP Growth_ algorithm to output a thoroughly developed, organized data structure of frequent itemsets. It is important to note that while this algorithm can be used to develop association rules, the data held within the FP tree and output from the FP growth algorithm are frequent itemsets and not association rules.
