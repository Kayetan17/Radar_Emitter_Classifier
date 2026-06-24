import numpy as np


def generate_pulse_offsets():
    print(np.random.normal(1000, 15, 8))  # PRI
    print(np.random.normal(1.0, 0.08, 8))  # pulse widths
    print(np.random.normal(9.0, 0.03, 8))  # frequency
    

generate_pulse_offsets()
    

#def pulse1(PRI, pulseW, frequency): 