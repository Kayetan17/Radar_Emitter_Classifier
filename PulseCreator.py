import numpy as np


def generate_pulse_offsets(pri_center, pw_center, freq_center, n_pulses=8):
    PRI = np.random.normal(pri_center, 90, n_pulses)  
    pulseW = np.random.normal(pw_center, 0.3, n_pulses)  
    frequency = np.random.normal(freq_center, 0.18, n_pulses)
    
    offset_pulse = np.stack([PRI, pulseW, frequency], axis=1)
    
    return offset_pulse


def generate_adaptive_train(pri, pw, freq, n_pulses=8, pri_shift=0.30, freq_shift=0.05):
    half = n_pulses // 2

    first = generate_pulse_offsets(pri, pw, freq, half)

    # Change the pulse at interval 4
    second = generate_pulse_offsets(pri * (1 + pri_shift), pw, freq * (1 + freq_shift), n_pulses - half)

    return np.vstack([first, second])


EMITTERS = {
    "Radar-A": {"pri": 1000, "pw": 1.0, "frequency": 9.0},
    "Radar-B": {"pri": 1150, "pw": 1.4, "frequency": 9.3},
    "Radar-C": {"pri":  850, "pw": 0.8, "frequency": 9.1},
    "Radar-D": {"pri": 1100, "pw": 1.2, "frequency": 9.2},
    "Radar-E": {"pri": 1300, "pw": 1.6, "frequency": 10},
}    



print("Adaptive train (first half normal ~1000 PRI, second half shifted ~1300 PRI):")
print(generate_adaptive_train(1000, 1.0, 9.0))