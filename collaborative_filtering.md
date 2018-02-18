- also referred to as _social filtering_
- filters information by using the recommendations of other people
- based on the idea that people who agreed in their evaluation of certain items in the past are likely to agree again in the future
- neighborhood-based approach
  - very popular solution
  - Prediction
    - number of users is selected based on their similarity to the active user, prediction for the active user is made by calculating a weighted average of the ratings of the selected users
    - instead of relying on the most similar person, a prediction is normally based on the weighted average of the recommendations of several people
    - weight given to a person's ratings is determined by the correlation between that person and the person for whom to make a prediction
    - as a measure of correlation the Pearson correlation coefficient can be used
  - selecting neighborhoods
    - when the number of users reaches a certain amount a selection of the best neighbors has to be made
    - two techniques: _correlation-thresholding_ and _best-n-neighbor_, can be used to determine which neighbors to select
    - _correlation-thresholding_ selects only those neighbors who's correlation is greater than a given threshold
    - _best-n-neighbor_ selects the best _n_ neighbors with the highest correlation
  - sparsity problem
    - too many ratings can be an issue, having to deal with too few ratings is a far more serious problem
    - _implicit ratings_: try to increase the number of ratings by inferring them from user's behavior
    - _dimensionality reduction_: by reducing the dimensionality of the information space, the ratings of two users can be used for predictions even if they did not rate the same time; matrix is projected into a lower dimensional space by using latent semantic indexing
    - _content description_: using the content of an item instead of the item itself could increase the amount of information people have in common
- Item-to-item approach
  - inversion of the neighborhood-based approach
  - instead of measuring the similarities between people the ratings are used to measure the correlation between items
  - Pearson correlation coefficient can again be used as a measure
- classification approach
  - movie can be directly represented as a vector, where each component of the vector corresponds to a rating of a different user
  - One-hot encoded entries, each user is represented by 2 rows each 1 for positive rating and 1 for negative and rated items are columns; each data point is a 0 or 1 indicating whether or not the given user positive or negative rating is set for that item



### Collaborative filtering-based recommendations

- idea behind collaborative filtering is to recommend new items based on the similarity of users
- measuring similarity
  - similarity can be calculated by _euclidean_ or _manhattan_ distance
  - we want a similarity measure between 0 and 1
- cosine similarity
  - most commonly used measure of similarity
  - with the cosine similarity, we are going to evaluate the similarity between two vectors based on the angle between them; smaller the angle, the more similar the two vectors are
  - maximum similarity when the angle between them is 0 (oriented in the same direction); 0 similarity when the angle between them is 90 (they are orthogonal to one another); -1 similarity when the angle between them is 180 (there are oriented in diametrically opposing directions)
  - if we restrict our vectors to non-negative values, then the angle of separation between the two vectors is bound between 0 and 90
- model-based collaborative filtering
  - problem set up to identify similar users:
    - we have an _nXm_ matrix consisting of the ratings of _n_ users and _m_ items; each element of the matrix _(i, j)_ represents how user _i_ rated item _j_
    - for each user, we want to recommend a set of movies they have not seen yet; we will effectively use an approach that is similar to weighted KNN
    - for each movie _j_ user _i_ has not seen yet, we find the set of users _U_ who are similar to user _i_ and have seen movie _j_; for each similar user _u_, we take _u_'s rating of a movie _j_ and multiply it by the cosine similarity of user _i_ and user _u_, sum these weighted ratings, divide by the number of users in _U_, and we get a weighted average rating for the movie _j_
    - sort movies by their weighted average rankings; these average rankings serve as an estimate for what hte user will rate each movie
  - ​

