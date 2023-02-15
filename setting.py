class Setting():
    def __init__(self):
        self.seed = 123
        self.pretrained = '/content/model1_20.pth'  # Pretrained model weights
        self.n_epochs = 30
        self.batch_size = 1

        ############################################################################################################
        self.PATHDATA = '/content/drive/MyDrive/coding_challenge'
        self.DATA_TRAIN = self.PATHDATA + '/train'
        self.DATA_VAL = self.PATHDATA + '/validation'
        self.DATA_TEST = self.PATHDATA + '/test'
        self.LOW_TRAIN = self.PATHDATA + '/low_train'
        self.LOW_VAL = self.PATHDATA + '/low_validation'
        self.LOW_TEST = self.PATHDATA + '/low_test'
        ############################################################################################################

        self.opt_hyp = {
            'adam': True,
            'nbs': 16,
            'lr0': 0.01,  # 0.01,
            'momentum': 0.87,
            'weight_decay': 0.00005,  # 0.0005,
        }

        self.Mirror = {
            'prob': 0.5,
        }


