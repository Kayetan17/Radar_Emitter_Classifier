import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from dataset import build_dataset
from features import extract_features
from PulseCreator import EMITTERS, generate_adaptive_train
import joblib

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


    X_cog = []
    y_cog = []
    for label_index, name in enumerate(EMITTERS):
        pri  = EMITTERS[name]["pri"]
        pw   = EMITTERS[name]["pw"]
        freq = EMITTERS[name]["frequency"]
        for _ in range(200):
            train = generate_adaptive_train(pri, pw, freq)
            X_cog.append(extract_features(train))
            y_cog.append(label_index)

    X_cog = np.array(X_cog)
    y_cog = np.array(y_cog)

    cog_predictions = clf.predict(X_cog)
    cog_acc = accuracy_score(y_cog, cog_predictions)
    print(f"\nCognitive radar accuracy: {cog_acc*100:.1f}%")
    print(f"  (was {acc*100:.1f}% on normal radars)")
    
    
    X_adapt_train = []
    y_adapt_train = []
    for label_index, name in enumerate(EMITTERS):
        pri  = EMITTERS[name]["pri"]
        pw   = EMITTERS[name]["pw"]
        freq = EMITTERS[name]["frequency"]
        for _ in range(200):
            train = generate_adaptive_train(pri, pw, freq)
            X_adapt_train.append(extract_features(train))
            y_adapt_train.append(label_index)

    X_adapt_train = np.array(X_adapt_train)
    y_adapt_train = np.array(y_adapt_train)

    X_robust = np.vstack([X_train, X_adapt_train])
    y_robust = np.concatenate([y_train, y_adapt_train])

    clf_robust = RandomForestClassifier(n_estimators=200, random_state=0)
    clf_robust.fit(X_robust, y_robust)

    robust_static_acc = accuracy_score(y_test, clf_robust.predict(X_test))
    robust_cog_acc    = accuracy_score(y_cog,  clf_robust.predict(X_cog))

    print(f"\n--- ROBUST MODEL (trained on steady + adaptive) ---")
    print(f"Static accuracy:    {robust_static_acc*100:.1f}%  (naive was {acc*100:.1f}%)")
    print(f"Cognitive accuracy: {robust_cog_acc*100:.1f}%  (naive was {cog_acc*100:.1f}%)")
    
    bundle = {
        "naive_model": clf,
        "robust_model": clf_robust,
        "radar_names": list(EMITTERS.keys()),
    }
    joblib.dump(bundle, "model.joblib")
    print("\nSaved both models to model.joblib")

if __name__ == "__main__":
    main()
    
