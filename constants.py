
# TUH EEG
TUH_CHANNELS = [
    "EEG FP1",
    "EEG FP2",
    "EEG F3",
    "EEG F4",
    "EEG C3",
    "EEG C4",
    "EEG P3",
    "EEG P4",
    "EEG O1",
    "EEG O2",
    "EEG F7",
    "EEG F8",
    "EEG T3",
    "EEG T4",
    "EEG T5",
    "EEG T6",
    "EEG FZ",
    "EEG CZ",
    "EEG PZ",
]

# Resampling frequency
TUH_FREQUENCY = 200
TUH_LABEL_DICT = {
    "fnsz": 0,
    "gnsz": 1,
    "spsz": 2,
    "cpsz": 3,
    "absz": 4,
    "tnsz": 5,
    "tcsz": 6,
    "mysz": 7,
}

# # Our SNUBH data - 8sensors
# DODH_CHANNELS = [
#     "C3-A2",
#     "C4-A1",
#     "F3-A2",
#     "F4-A1",
#     "O1-A2",
#     "O2-A1",
#     "LOC",
#     "ROC"
# ]
# DREEM_FREQ = 500
# DREEM_LABEL_DICT = \
# {
#     0:"WAKE", 
#     1: "N1", 
#     2: "N2", 
#     3: "N3", 
#     4: "REM"
# }

# Our SNUBH data - 10sensors
OURS_CHANNELS = [
    "C3-A2",
    "C4-A1",
    "F3-A2",
    "F4-A1",
    "O1-A2",
    "O2-A1",
    "LOC",
    "ROC",
    "EKG",
    "CHIN EMG"
]
DREEM_FREQ = 500
DREEM_LABEL_DICT = \
{
    0:"WAKE", 
    1: "N1", 
    2: "N2", 
    3: "N3", 
    4: "REM"
}

# Original Dreem DOD-H 
DODH_CHANNELS = [
    "C3_M2",
    "F3_F4",
    "F3_M2",
    "F3_O1",
    "F4_M1",
    "F4_O2",
    "FP1_F3",
    "FP1_M2",
    "FP1_O1",
    "FP2_F4",
    "FP2_M1",
    "FP2_O2",
    "ECG",
    "EMG",
    "EOG1",
    "EOG2",
]
DREEM_FREQ = 250
DREEM_LABEL_DICT = \
{
    0:"WAKE", 
    1: "N1", 
    2: "N2", 
    3: "N3", 
    4: "REM"
}