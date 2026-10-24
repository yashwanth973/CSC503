# This file is for implementing the K-Nearest Neighbor model
import pandas as pd


def jaccard_distance(o1, o2):
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

def knn(O_star, o_, k, d):
    if (k <= 0 ):
        raise ValueError("The provided value k must be positive")

    distances = []

    for student_features, major in O_star:
        distance = d(o_, student_features)
        distances.append((distance, major))

    distances.sort(key=lambda x:x[0]) #Sorts observation by distance in ascending order
    neighbors = distances[:k]
    counts = {}
    
    #
    for distance, major in neighbors:
        counts[major] = counts.get(major, 0) + 1

    predicted_major = max(counts, key=counts.get)

    return predicted_major

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


#Temporary Testing KNN Model Code:
training_data, evaluation_data = process_file()

correct_predictions = 0
incorrect_predictions = 0

for o_ in evaluation_data:

    prediction = knn(
        training_data,
        o_,
        k=50,
        d=jaccard_distance
    )

    print("Recommended major:", prediction)
    print("True major:", o_[1])
    if prediction == o_[1]:
        print("Prediction Correct!")
        correct_predictions += 1
    else:
        print("Prediction Incorrect =(")
        incorrect_predictions += 1

print(f"----- SUMMARY -----")
print(f"Number of Correct Predictions: {correct_predictions}")
print(f"Number of Incorrect Predictions: {incorrect_predictions}")
print(f"Accuracy: {(correct_predictions / len(evaluation_data)) * 100}")