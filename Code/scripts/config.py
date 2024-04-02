class TrainConfig_1(object):
    learning_rate_1 = 0.00001
    batch_size_1 = 50
    n_epoch_1 = 50

class TrainConfig_2(object):
    learning_rate_2 = 0.00001
    batch_size_2 = 1
    n_epoch_2 = 50


class DataConfig(object):
    fs = 50
    fs_sub = 50
    SNR = 20
    n_classes = 3 #this is specific for classification. # 1: Rotor/no rotor ; 2: RA/LA/No rotor (2 classes) ; 3: 7 regions (3 classes) + no rotor (8 classes)
    DF_Mapping = False
