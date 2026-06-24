import numpy as np


def generate_pulse_offsets(pri_center, pw_center, freq_center, n_pulses):
    PRI = np.random.normal(pri_center, 15, n_pulses)  
    pulseW = np.random.normal(pw_center, 0.08, n_pulses)  
    frequency = np.random.normal(freq_center, 0.03, n_pulses)
    
    offset_pulse = np.stack([PRI, pulseW, frequency], axis=1)
    
    return offset_pulse
    

print(generate_pulse_offsets(1000, 1, 9))
print(generate_pulse_offsets(2000, 2, 10))

    

#def pulse1(PRI, pulseW, frequency): 