from Models.multinomial_naive_bayes_classifier import MultinomialNaiveBayes

MAX_RESULTS = 2

def main():
    # Multinomia Naive Bayes Classifier
    print(f"Starting Multinomial Naive Bayes Classifier...")

    MNB_classifier = MultinomialNaiveBayes()
    training_data, evaluation_data = MNB_classifier.process_file()
    
    MNB_classifier.fit(training_data)
    student_features, actual_major = evaluation_data[0]

    predictions = MNB_classifier.predict(student_features, num_results=MAX_RESULTS)

    print(f"Prediction results for MNB: {predictions}")


if __name__ == "__main__":
    main()