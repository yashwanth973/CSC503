import pandas as pd

class MultinomialNaiveBayes:


    def predict():
        return 

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
