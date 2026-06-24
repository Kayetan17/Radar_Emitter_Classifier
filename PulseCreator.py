import numpy as np


def generate_pulse_offsets(pri_center, pw_center, freq_center, n_pulses=8):
    PRI = np.random.normal(pri_center, 90, n_pulses)  
    pulseW = np.random.normal(pw_center, 0.3, n_pulses)  
    frequency = np.random.normal(freq_center, 0.18, n_pulses)
    
    offset_pulse = np.stack([PRI, pulseW, frequency], axis=1)
    
    return offset_pulse


EMITTERS = {
    "Radar-A": {"pri": 1000, "pw": 1.0, "frequency": 9.0},
    "Radar-B": {"pri": 1150, "pw": 1.4, "frequency": 9.3},
    "Radar-C": {"pri":  850, "pw": 0.8, "frequency": 9.1},
    "Radar-D": {"pri": 1100, "pw": 1.2, "frequency": 9.2},
    "Radar-E": {"pri": 1300, "pw": 1.6, "frequency": 10},
}    



    
