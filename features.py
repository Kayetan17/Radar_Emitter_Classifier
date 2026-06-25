import numpy as np

def extract_features(train):
    pri  = train[:, 0]   # all 8 PRI values
    pw   = train[:, 1]   # all 8 pulse-width values
    frequency = train[:, 2]   # all 8 frequency values
    
    half = max(1, len(pri) // 2)
    pri_drift  = abs(pri[half:].mean()  - pri[:half].mean())
    freq_drift = abs(frequency[half:].mean() - frequency[:half].mean())

    # build a list of summary numbers from pri, pw, frequency
    features = [
        pri.mean(),
        pw.mean(),
        frequency.mean(),
        pri.std(),
        pw.std(),
        frequency.std(),
        pri_drift,
        freq_drift
        
    ]
    return np.array(features)


if __name__ == "__main__":
    from PulseCreator import generate_pulse_offsets, EMITTERS
    train = generate_pulse_offsets(1000, 1.0, 9.0)
    print(extract_features(train))