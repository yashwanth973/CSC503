from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd

class MultinomialNaiveBayes:

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def predict(self):
        return 

    @staticmethod
    def process_file(self):
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

#Extracting the features and majors from trainnig data set
X_train = [features for features, major in training_data]
y_train = [major for features, major in training_data]

#Extracting the features and majors from evaluation data set
X_test = [features for features, major in evaluation_data]
y_test = [major for features, major in evaluation_data]

model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)