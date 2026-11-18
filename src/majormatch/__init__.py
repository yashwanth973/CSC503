"""MajorMatch: the proposed recommenders (k-NN with Jaccard distance, Multinomial Naive Bayes)."""
from src.majormatch.knn import KNN, jaccard_distance
from src.majormatch.mnb import MultinomialNaiveBayes

__all__ = ["KNN", "jaccard_distance", "MultinomialNaiveBayes"]
