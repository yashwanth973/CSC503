from Models.multinomial_naive_bayes_classifier import MultinomialNaiveBayes
from Models.knn_classifier import KNN
MAX_RESULTS = 2

def main():
    # Multinomia Naive Bayes Classifier
    print(f"Starting Multinomial Naive Bayes Classifier...")

    MNB_classifier = MultinomialNaiveBayes()
    training_data, evaluation_data = MNB_classifier.process_file()
    
    MNB_classifier.fit(training_data)
    student_features, actual_major = evaluation_data[0]

    predictions = MNB_classifier.predict(student_features, num_results=MAX_RESULTS)

    print(f"Prediction results for MNB: {[ major for major, score in predictions]}")
    
    #K-NN Classifier
    print("Starting K-Nearest Neighbor Classifier")
    K = 7
    
    # Load and prepare the data once, not on every call
    model = KNN()
    training_data, evaluation_data = model.process_file()


    def recommend_majors(selected, top_n=2, k=K):
        """
        selected: list of feature names the student has, e.g. ["Programming", "Coding Club"]
        returns: list of the top_n recommended majors
        """
        unknown = [s for s in selected if s not in KNN.FEATURES]
        if unknown:
            raise ValueError(f"Unknown features: {unknown}")
    
        # Build the 12-length binary vector in the same order as training
        query = [1 if f in selected else 0 for f in KNN.FEATURES]
    
        return model.knn(
            training_data,
            query,
            k=k,
            d=model.jaccard_distance,
            n=top_n,
        )


    def evaluate(top_n=2, k=K):
        hits = 0
        for features, true_major in evaluation_data:
            if true_major in model.knn(training_data, features, k=k,
                                       d=model.jaccard_distance, n=top_n):
                hits += 1
        return hits / len(evaluation_data)


    print(recommend_majors(["Programming", "Data Analysis", "Coding Club"]))
    print(f"Top-1 accuracy: {evaluate(top_n=1):.2%}")
    print(f"Top-2 accuracy: {evaluate(top_n=2):.2%}")

if __name__ == "__main__":
    main()
