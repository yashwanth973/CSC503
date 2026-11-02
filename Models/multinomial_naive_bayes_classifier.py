from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from collections import Counter
import pandas as pd
import numpy as np
import math

class MultinomialNaiveBayes:

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_priors = {}
        self.feature_probabilities = {}

    def fit(self, training_data):
        major_count = Counter(major for features, major in training_data)
        total = len(training_data)
        num_features = len(training_data[0][0])

        for major, count in major_count.items():
            print(F"major: {major}, count: {count}")
            self.class_priors[major] = count / total

        feature_counts = {
            major: [0] * num_features
            for major in major_count
        }

        for features, major in training_data:
            for i, value in enumerate(features):
                feature_counts[major][i] += value

        for major in major_count:

            total_feature_count = sum(feature_counts[major])

            self.feature_probabilities[major] = []

            for count in feature_counts[major]:
                #Laplace Smoothing
                probability = (
                    (count + self.alpha)
                    / (total_feature_count + self.alpha * num_features)
                )

                self.feature_probabilities[major].append(probability)

    def predict(self, student_features):
        best_major = None
        best_score = float("-inf")

        for major, prior in self.class_priors.items():

            # Start with the log of the class prior
            score = math.log(prior)

            # Add the contribution of each feature
            for feature_value, probability in zip(
                student_features,
                self.feature_probabilities[major]
            ):
                score += feature_value * math.log(probability)

            # Keep track of the highest-scoring major
            if score > best_score:
                best_score = score
                best_major = major

        return best_major

    @staticmethod
    def process_file():
        df = pd.read_csv(r"Datasets\student_data_extended.xlsx - Organized Data.csv", header = 1)

        features = [
            "Artistic",
            "Data Analysis",
            "Leadership",
            "Problem Solving",
            "Programming",
            "Public Speaking",
            "Art Club",
            "Coding Club",
            "Debate Club",
            "Music Club",
            "Sports Team",
            "Volunteer Group"
        ]

        training_df = df.iloc[:700]
        evaluation_df= df.iloc[700:]
        print("Training data:", training_df.shape)
        print("Evaluation data:", evaluation_df.shape)

        training_data = []
        evaluation_data = []

        for index, row in training_df.iterrows():
            student_features = row[features].astype(int).tolist()
            major = row["Major"]

            training_data.append((student_features, major))
            

        for index, row in evaluation_df.iterrows():
            student_features = row[features].astype(int).tolist()
            major = row["Major"]

            evaluation_data.append((student_features, major))

        return training_data, evaluation_data

MNB_classifier = MultinomialNaiveBayes()

training_data, evaluation_data = MNB_classifier.process_file()

MNB_classifier.fit(training_data)

#Extracting the features and majors from trainnig data set
X_train = [features for features, major in training_data]
y_train = [major for features, major in training_data]

#Extracting the features and majors from evaluation data set
X_test = [features for features, major in evaluation_data]
y_test = [major for features, major in evaluation_data]

print("Training distribution:", Counter(y_train))
print("Evaluation distribution:", Counter(y_test))

our_predictions = [MNB_classifier.predict(features) for features, major in evaluation_data]

our_accuracy = accuracy_score(y_test, our_predictions)

print("Our MNB accuracy:", our_accuracy)
