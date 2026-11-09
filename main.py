from Models.multinomial_naive_bayes_classifier import MultinomialNaiveBayes
from Models.knn_classifier import KNN
MAX_RESULTS = 2
K = 7

def main():

    query = convert_data(["Programming", "Data Analysis", "Coding Club"])

    # Multinomia Naive Bayes Classifier
    print(f"Starting Multinomial Naive Bayes Classifier...")

    MNB_classifier = MultinomialNaiveBayes()
    training_data, evaluation_data = MNB_classifier.process_file()
    
    MNB_classifier.fit(training_data)

    predictions = MNB_classifier.predict(query, num_results=MAX_RESULTS)
    mnb_results = [ major for major, score in predictions]

    #K-NN Classifier
    print("Starting K-Nearest Neighbor Classifier")
    # Load and prepare the data once, not on every call
    model = KNN()

    knn_results = model.knn(
            training_data,
            query,
            k=K,
            d=model.jaccard_distance,
            n=MAX_RESULTS,
        )

    print(f"Prediction results for MNB: {mnb_results}")
    print(f"Prediction results for KNN: {knn_results}")
    print(f"Top-1 accuracy: {evaluate(evaluation_data,training_data,top_n=1):.2%}")
    print(f"Top-2 accuracy: {evaluate(evaluation_data,training_data,top_n=2):.2%}")


def convert_data(selected):
        """
        selected: list of feature names the student has, e.g. ["Programming", "Coding Club"]
        returns: list of the top_n recommended majors
        """
        unknown = [s for s in selected if s not in KNN.FEATURES]
        if unknown:
            raise ValueError(f"Unknown features: {unknown}")
    
        # Build the 12-length binary vector in the same order as training
        query = [1 if f in selected else 0 for f in KNN.FEATURES]
    
        return query

def evaluate(evaluation_data, training_data, top_n=2, k=K ):
    hits = 0
    model = KNN()
    for features, true_major in evaluation_data:
        if true_major in model.knn(training_data, features, k=k,
                                    d=model.jaccard_distance, n=top_n):
            hits += 1
    return hits / len(evaluation_data)


if __name__ == "__main__":
    main()
