# This file is for implementing the K-Nearest Neighbor model

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

    for distance, major in neighbors:
        counts[major] = counts.get(major, 0) + 1

    predicted_major = max(counts, key=counts.get)

    return predicted_major




#Temporary Testing KNN Model Code:
training_data = [
    ([1, 1, 1, 0], "Computer Science"),
    ([1, 1, 0, 0], "Computer Science"),
    ([0, 0, 1, 1], "Biology"),
    ([0, 1, 0, 1], "Mathematics"),
    ([1, 0, 1, 0], "Computer Science")
]

new_observation = [1, 1, 1, 0]

prediction = knn(
    training_data,
    new_observation,
    k=3,
    d=jaccard_distance
)

print("Recommended major:", prediction)