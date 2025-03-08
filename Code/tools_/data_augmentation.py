import sys
sys.path.append("../Code")

import random
import numpy as np
import matplotlib.pyplot as plt

from scripts.config import ParseHiperparams

class DataAugmentation:
    def __init__(self, params, data="toy"):
        self.params=params
        self.data=data

    def time_masking(self ):
        '''
        This function masks some parts of the the signal to 0 in a batch-wise fashion
        
        '''
        if self.data=="toy":
            self.data = np.random.randint(low=0, high=256, size=(221, 400, 12, 32, 1))  
        mask_ones= np.ones(self.data.shape)
        
        percentage_of_signal = 0.2
        window_len = int(percentage_of_signal*self.params["batch_size"])-1

        for n_batch in range(self.data.shape[0]):

            starting_index=random.randint(0,self.params["batch_size"]-window_len)
            mask_ones[n_batch, starting_index : starting_index + window_len, :, :] = 0

            
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(mask_ones[n_batch, :, 0, 0, 0])
        plt.subplot(2, 1, 2)
        plt.plot(mask_ones[n_batch, :, 1, 1, 0])
        plt.savefig('output/figures/evaluation_trash/augmentation.png')
        print('output/figures/evaluation_trash/augmentation.png')
        plt.close()

        masked_data=self.data * mask_ones

        
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(masked_data[67, :, 0, 0, 0])
        plt.subplot(2, 1, 2)
        plt.plot(masked_data[67, :, 1, 1, 0])
        plt.savefig('output/figures/evaluation_trash/augmentation.png')
        print('output/figures/evaluation_trash/augmentation.png')
        plt.close()

        return masked_data


if __name__ == "__main__":

    params = ParseHiperparams().parse_default_hyperparams()
    DataAugmentation_obj=DataAugmentation(params)
    DataAugmentation_obj.time_masking()  
        