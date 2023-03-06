window_size = 2048

n_way = 10
batch_size = 32
best = -1
evaluate_every = 200
loss_every = 20
n_iter = 1500
learning_rate = 0.001
n_val = 2
n = 0
save_weights_file = "weights-best-10-oneshot-low-data.hdf5"


exps_idx = {
    '12DriveEndFault':0,
    '12FanEndFault':9,
    '48DriveEndFault':0
}

faults_idx = {
    'Normal': 0,
    '0.007-Ball': 1,
    '0.014-Ball': 2,
    '0.021-Ball': 3,
    '0.007-InnerRace': 4,
    '0.014-InnerRace': 5,
    '0.021-InnerRace': 6,
    '0.007-OuterRace6': 7,
    '0.014-OuterRace6': 8,
    '0.021-OuterRace6': 9,
}