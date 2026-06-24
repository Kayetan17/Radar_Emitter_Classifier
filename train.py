import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from dataset import build_dataset


def main():
    X, y = build_dataset(samples_per_radar=400)
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {acc*100:.1f}%")
    
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(y_test, predictions))

    print("\nDetailed report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()