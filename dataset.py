from PulseCreator import generate_pulse_offsets, EMITTERS
from features import extract_features



def build_dataset(samples_per_radar=400):
    X = []   # the pulse trains
    y = []   # the matching labels (integers)

    for label_index, name in enumerate(EMITTERS):
        pri  = EMITTERS[name]["pri"]
        pw   = EMITTERS[name]["pw"]
        frequency = EMITTERS[name]["frequency"]

        for _ in range(samples_per_radar):
            train = generate_pulse_offsets(pri, pw, frequency)
            X.append(extract_features(train))
            y.append(label_index)

    return X, y


if __name__ == "__main__":
    X, y = build_dataset(samples_per_radar=50)
    print("Number of trains:", len(X))
    print("Number of labels:", len(y))
    print("First train shape:", X[0].shape)
    print("First 5 labels:", y[:5])
    print("Last 5 labels:", y[-5:])