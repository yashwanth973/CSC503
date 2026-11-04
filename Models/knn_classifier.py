import pandas as pd
import numpy as np

filename = "Datasets/student_data_extended.xlsx - Organized Data.csv"

class KNN:

    FEATURES = [
        "Artistic", "Data Analysis", "Leadership", "Problem Solving",
        "Programming", "Public Speaking", "Art Club", "Coding Club",
        "Debate Club", "Music Club", "Sports Team", "Volunteer Group"
    ]

    def jaccard_distance(self, o1, o2):
        intersection = 0
        union = 0

        for a,b in zip(o1,o2):
            if a == 1 and b == 1:
                intersection +=1
            if a == 1 or b == 1:
                union += 1
        if union == 0:
            return 0.0

        return 1 - (intersection/union)

    def knn(self, O_star, o_, k, d, n):
        if (k <= 0 ):
            raise ValueError("The provided value k must be positive")

        distances = []

        for student_features, major in O_star:
            distance = d(o_, student_features)
            distances.append((distance, major))

        distances.sort(key=lambda x:x[0]) #Sorts observation by distance in ascending order
        neighbors = distances[:k]
        counts = {}
        closest = {}

        #
        for distance, major in neighbors:
            counts[major] = counts.get(major, 0) + 1
            if major not in closest:
              closest[major] = distance
        ranked = sorted(counts, key = lambda m: (-counts[m], closest[m])) # if the votes are tie then compare the distance
        if n == 1:
          return ranked[0]

        return ranked[:n]

    def process_file(self):
        df = pd.read_csv(filename, header = 1)

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
            student_features = row[self.FEATURES].astype(int).tolist()
            major = row["Major"]

            training_data.append((student_features, major))


        for index, row in evaluation_df.iterrows():
            student_features = row[features].astype(int).tolist()
            major = row["Major"]

            evaluation_data.append((student_features, major))

        return training_data, evaluation_data
