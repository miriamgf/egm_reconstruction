import math
from sklearn.model_selection import StratifiedShuffleSplit



class KFold_Stratified:
    '''
    Class to compute k-folds for the dataset. 
    The dataset is stratified according to 3 groups: sinusal, simple rotor and complex rotor rythms
    
    '''
    def __init__(self, k, train_size = 0.7, test_size = 0.2, val_size = 0.1):

        self.k = k
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        
        self.sinusal_groups = {
            "Simulation_01_200316_001_7", "Simulation_01_210209_001_002", 
            "Simulation_01_210205_001_002", "Simulation_01_200428_001_001", 
            "Simulation_01_200316_001_5", "Simulation_01_200428_001_005", 
            "Simulation_01_200212_001_7", "Simulation_01_210119_001_001", 
            "Simulation_01_210205_001_003", "Simulation_01_201223_001_002", 
            "Simulation_01_200212_001_1", "Simulation_01_200428_001_003", 
            "Simulation_01_200316_001_3", "Simulation_01_200316_001_4", 
            "Sinusal_150629", "RA_RAA_141216", "Simulation_01_200428_001_002", 
            "Simulation_01_210210_001_001", "Simulation_01_210208_001_002", 
            "Simulation_01_200212_001_9", "Simulation_01_200316_001_1", 
            "Simulation_01_200212_001_6", "Simulation_01_200212_001_5", 
            "Simulation_01_210209_001_003", "Simulation_01_200316_001_9", 
            "Simulation_01_210209_001_001", "Simulation_01_200428_001_006", 
            "Simulation_01_200316_001_5",
        }

        self.rotor_simple_groups = {
            "Simulation_01_200212_001_2", "Simulation_01_191001_001_002", 
            "Simulation_01_190717_001_001", "Simulation_01_191001_001_007", 
            "Simulation_01_200316_001_8", "Simulation_01_200428_001_004", 
            "Simulation_01_200212_001_8", "Simulation_01_200212_001_10", 
            "Simulation_01_191001_001_001", "Simulation_01_200428_001_007", 
            "Simulation_01_200428_001_009", "Simulation_01_200428_001_010", 
            "Simulation_01_200316_001_2", "Simulation_01_200316_001_6", 
            "Simulation_01_200316_001_10", "Simulation_01_190502_001_003", 
            "Simulation_01_190619_001_002", "Simulation_01_191001_001_005", 
            "Simulation_01_200212_001_4", "Simulation_01_200212_001_10",
            "Simulation_01_200428_001_008", "Simulation_01_190502_001_005"
        }

        self.rotor_complejo_groups = {
            "LA_RSPV_CAF_150115", "LA_RSPV_150113", "LA_PLAW_140612", 
            "LA_LSPV_150203", "LA_LSPV_150113", "RA_RAFW_140807", 
            "RA_RAA_141230", "TwoRotors_181219", "LA_PLAW_140711_arm", 
            "RA_RAFW_SAF_140730", "LA_RIPV_150121", "LA_LIPV_150119"
        } 
        
    def compute_k_folds(self):

        '''
        This function computes the k-folds for the dataset.
        '''


        all_data = list(self.sinusal_groups | self.rotor_simple_groups | self.rotor_complejo_groups)
        labels = (["sinusal"] * len(self.self.sinusal_groups) +
                ["simple_rotor"] * len(self.rotor_simple_groups) +
                ["rotor_complejo"] * len(self.rotor_complejo_groups))
    
        sss = StratifiedShuffleSplit(n_splits=self.k, test_size=(self.test_size + self.val_size), random_state=42)

        self.folds_list = {}
        fold = 0
        for train_idx, val_test_idx in sss.split(all_data, labels):
            train = [all_data[i] for i in train_idx]
            val_test = [all_data[i] for i in val_test_idx]
            val_test_labels = [labels[i] for i in val_test_idx]
            
            sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size/(self.test_size + self.val_size), random_state=42)
            
            for val_idx, test_idx in sss_val_test.split(val_test, val_test_labels):
                val = [val_test[i] for i in val_idx]
                test = [val_test[i] for i in test_idx]
                
                # Calcular las proporciones exactas
                total_data = len(all_data)
                total_train = math.ceil(total_data * self.train_size)  
                remaining_data = total_data - total_train         

                total_val = math.ceil(remaining_data * self.val_size / (self.val_size + self.test_size))  
                total_test = remaining_data - total_val  
                
                self.folds_list[f"fold_{fold}"] = {
                    "train": train[:total_train], 
                    "val": val[:total_val], 
                    "test": test[:total_test]
                }

                fold += 1
        return self.folds_list
    
    def select_fold(self, fold):

        '''
        This function returns the train, val and test data for a given fold.
        The data is precomputed and stored in a dictionary.
        '''
        precomputed_folds = {
            'fold_0': {'train': ['Simulation_01_191001_001_002',
            'Simulation_01_200316_001_5',
            'Simulation_01_190502_001_003',
            'Simulation_01_190717_001_001',
            'RA_RAA_141230',
            'Simulation_01_200212_001_10',
            'Simulation_01_200428_001_006',
            'LA_PLAW_140612',
            'Simulation_01_200428_001_001',
            'Simulation_01_200316_001_7',
            'Simulation_01_200212_001_8',
            'Sinusal_150629',
            'Simulation_01_200212_001_6',
            'Simulation_01_200316_001_9',
            'Simulation_01_210209_001_001',
            'Simulation_01_200428_001_002',
            'RA_RAFW_140807',
            'Simulation_01_200316_001_4',
            'Simulation_01_190502_001_005',
            'Simulation_01_200428_001_004',
            'LA_RSPV_CAF_150115',
            'Simulation_01_191001_001_001',
            'Simulation_01_200428_001_008',
            'LA_RIPV_150121',
            'Simulation_01_210205_001_003',
            'Simulation_01_200212_001_9',
            'Simulation_01_210209_001_002',
            'Simulation_01_200212_001_1',
            'Simulation_01_201223_001_002',
            'Simulation_01_200428_001_007',
            'Simulation_01_191001_001_007',
            'Simulation_01_200428_001_005',
            'LA_LSPV_150203',
            'LA_LSPV_150113',
            'RA_RAFW_SAF_140730',
            'Simulation_01_210208_001_002',
            'Simulation_01_200428_001_009',
            'Simulation_01_200428_001_010',
            'Simulation_01_210209_001_003',
            'LA_PLAW_140711_arm',
            'Simulation_01_200316_001_8'],
            'val': ['Simulation_01_210210_001_001',
            'Simulation_01_200316_001_6',
            'TwoRotors_181219',
            'Simulation_01_191001_001_005',
            'Simulation_01_210205_001_002',
            'Simulation_01_200316_001_1'],
            'test': ['Simulation_01_200316_001_3',
            'Simulation_01_200212_001_7',
            'RA_RAA_141216',
            'Simulation_01_200212_001_4',
            'Simulation_01_190619_001_002',
            'LA_RSPV_150113',
            'Simulation_01_200212_001_2',
            'Simulation_01_200212_001_5',
            'Simulation_01_200316_001_2',
            'Simulation_01_210119_001_001',
            'LA_LIPV_150119',
            'Simulation_01_200428_001_003']},
            'fold_1': {'train': ['LA_RIPV_150121',
            'Simulation_01_200316_001_4',
            'Simulation_01_200428_001_009',
            'Simulation_01_191001_001_007',
            'LA_RSPV_CAF_150115',
            'Simulation_01_200316_001_3',
            'Simulation_01_200428_001_003',
            'Simulation_01_200428_001_008',
            'LA_LSPV_150203',
            'Simulation_01_200428_001_005',
            'Simulation_01_200428_001_010',
            'Simulation_01_200316_001_6',
            'Simulation_01_191001_001_002',
            'Simulation_01_200212_001_2',
            'LA_PLAW_140612',
            'Simulation_01_200212_001_8',
            'Simulation_01_210209_001_003',
            'Simulation_01_210208_001_002',
            'Simulation_01_190502_001_005',
            'Simulation_01_200212_001_1',
            'Simulation_01_200316_001_7',
            'Simulation_01_210119_001_001',
            'Simulation_01_200212_001_7',
            'Simulation_01_191001_001_005',
            'Simulation_01_200212_001_9',
            'RA_RAA_141216',
            'Simulation_01_200428_001_001',
            'LA_RSPV_150113',
            'Simulation_01_200316_001_9',
            'LA_PLAW_140711_arm',
            'Simulation_01_210205_001_003',
            'Simulation_01_200316_001_2',
            'Simulation_01_190619_001_002',
            'Simulation_01_200428_001_007',
            'LA_LIPV_150119',
            'Simulation_01_200212_001_10',
            'RA_RAFW_SAF_140730',
            'Simulation_01_200428_001_002',
            'Sinusal_150629',
            'Simulation_01_190502_001_003',
            'Simulation_01_201223_001_002'],
            'val': ['Simulation_01_200316_001_10',
            'Simulation_01_210209_001_001',
            'Simulation_01_200316_001_5',
            'Simulation_01_200212_001_5',
            'Simulation_01_210205_001_002',
            'TwoRotors_181219'],
            'test': ['RA_RAFW_140807',
            'Simulation_01_190717_001_001',
            'Simulation_01_200428_001_004',
            'Simulation_01_200212_001_4',
            'RA_RAA_141230',
            'LA_LSPV_150113',
            'Simulation_01_210209_001_002',
            'Simulation_01_200316_001_1',
            'Simulation_01_200428_001_006',
            'Simulation_01_210210_001_001',
            'Simulation_01_200212_001_6',
            'Simulation_01_200316_001_8']},
            'fold_2': {'train': ['Simulation_01_200428_001_010',
            'Simulation_01_200428_001_005',
            'RA_RAA_141230',
            'LA_RIPV_150121',
            'Simulation_01_210210_001_001',
            'Simulation_01_200316_001_3',
            'LA_LSPV_150113',
            'Simulation_01_200428_001_004',
            'Simulation_01_190717_001_001',
            'Simulation_01_200428_001_009',
            'Simulation_01_200428_001_001',
            'Simulation_01_201223_001_002',
            'Simulation_01_210205_001_002',
            'Simulation_01_210209_001_001',
            'Simulation_01_200316_001_4',
            'Simulation_01_190502_001_005',
            'Simulation_01_200316_001_10',
            'Simulation_01_210119_001_001',
            'Simulation_01_200428_001_003',
            'Sinusal_150629',
            'TwoRotors_181219',
            'Simulation_01_200428_001_007',
            'Simulation_01_200212_001_4',
            'Simulation_01_210208_001_002',
            'LA_LIPV_150119',
            'Simulation_01_210209_001_002',
            'LA_PLAW_140612',
            'Simulation_01_200212_001_2',
            'Simulation_01_190502_001_003',
            'Simulation_01_190619_001_002',
            'Simulation_01_200212_001_7',
            'Simulation_01_200212_001_5',
            'RA_RAA_141216',
            'Simulation_01_200316_001_9',
            'RA_RAFW_140807',
            'LA_RSPV_150113',
            'Simulation_01_200212_001_8',
            'Simulation_01_210205_001_003',
            'Simulation_01_191001_001_007',
            'Simulation_01_200212_001_10',
            'Simulation_01_200316_001_5'],
            'val': ['Simulation_01_200316_001_7',
            'Simulation_01_200428_001_008',
            'Simulation_01_200316_001_1',
            'Simulation_01_200212_001_1',
            'Simulation_01_200316_001_6',
            'Simulation_01_200212_001_9'],
            'test': ['Simulation_01_191001_001_005',
            'Simulation_01_200316_001_2',
            'LA_PLAW_140711_arm',
            'RA_RAFW_SAF_140730',
            'LA_RSPV_CAF_150115',
            'Simulation_01_200316_001_8',
            'Simulation_01_200428_001_006',
            'Simulation_01_191001_001_002',
            'Simulation_01_210209_001_003',
            'Simulation_01_200428_001_002',
            'Simulation_01_200212_001_6',
            'LA_LSPV_150203']},
            'fold_3': {'train': ['LA_LSPV_150203',
            'Simulation_01_200428_001_003',
            'LA_LIPV_150119',
            'Simulation_01_200212_001_5',
            'Simulation_01_190619_001_002',
            'LA_PLAW_140711_arm',
            'Simulation_01_210209_001_003',
            'Simulation_01_191001_001_002',
            'RA_RAA_141230',
            'LA_RIPV_150121',
            'Simulation_01_191001_001_005',
            'Simulation_01_200212_001_2',
            'Simulation_01_210119_001_001',
            'Simulation_01_200316_001_3',
            'Simulation_01_200428_001_001',
            'Simulation_01_200316_001_9',
            'Simulation_01_210208_001_002',
            'Simulation_01_200316_001_10',
            'Simulation_01_200316_001_8',
            'Simulation_01_200212_001_10',
            'Simulation_01_200428_001_004',
            'RA_RAA_141216',
            'Simulation_01_191001_001_001',
            'Simulation_01_200316_001_7',
            'Simulation_01_210210_001_001',
            'Simulation_01_201223_001_002',
            'TwoRotors_181219',
            'Simulation_01_210209_001_002',
            'Simulation_01_190502_001_003',
            'Simulation_01_200316_001_6',
            'Simulation_01_200316_001_2',
            'Simulation_01_200316_001_4',
            'Simulation_01_200428_001_010',
            'Simulation_01_200428_001_005',
            'LA_RSPV_150113',
            'Simulation_01_191001_001_007',
            'Simulation_01_200212_001_1',
            'Simulation_01_200316_001_1',
            'LA_PLAW_140612',
            'LA_LSPV_150113',
            'Simulation_01_200212_001_8'],
            'val': ['Simulation_01_200212_001_4',
            'Simulation_01_200428_001_006',
            'Simulation_01_200212_001_9',
            'Simulation_01_200316_001_5',
            'Simulation_01_210209_001_001',
            'Simulation_01_210205_001_003'],
            'test': ['Simulation_01_190502_001_005',
            'Simulation_01_210205_001_002',
            'Simulation_01_200428_001_008',
            'RA_RAFW_SAF_140730',
            'LA_RSPV_CAF_150115',
            'Simulation_01_200428_001_007',
            'Simulation_01_190717_001_001',
            'Sinusal_150629',
            'Simulation_01_200212_001_6',
            'Simulation_01_200428_001_009',
            'Simulation_01_200212_001_7',
            'RA_RAFW_140807']},
            'fold_4': {'train': ['Simulation_01_210205_001_003',
            'Simulation_01_200316_001_3',
            'Simulation_01_200428_001_009',
            'Simulation_01_191001_001_002',
            'Simulation_01_200316_001_6',
            'Simulation_01_200212_001_9',
            'Simulation_01_200316_001_1',
            'Simulation_01_200428_001_006',
            'Simulation_01_190717_001_001',
            'Simulation_01_200316_001_9',
            'Simulation_01_200428_001_005',
            'Simulation_01_200212_001_2',
            'TwoRotors_181219',
            'LA_LIPV_150119',
            'Simulation_01_191001_001_001',
            'Simulation_01_200428_001_008',
            'Simulation_01_210208_001_002',
            'LA_PLAW_140612',
            'Simulation_01_210209_001_003',
            'Simulation_01_190619_001_002',
            'LA_LSPV_150203',
            'Simulation_01_200316_001_5',
            'Simulation_01_210119_001_001',
            'LA_PLAW_140711_arm',
            'Simulation_01_200212_001_8',
            'Simulation_01_200316_001_8',
            'Simulation_01_200316_001_4',
            'LA_RIPV_150121',
            'Simulation_01_200212_001_10',
            'Simulation_01_200316_001_7',
            'Simulation_01_200212_001_7',
            'Simulation_01_200212_001_5',
            'RA_RAA_141230',
            'Simulation_01_200428_001_001',
            'Simulation_01_200428_001_010',
            'Simulation_01_200316_001_2',
            'RA_RAFW_140807',
            'Simulation_01_200316_001_10',
            'Simulation_01_190502_001_003',
            'Simulation_01_210205_001_002',
            'Simulation_01_190502_001_005'],
            'val': ['Simulation_01_210210_001_001',
            'Simulation_01_210209_001_002',
            'Simulation_01_200212_001_1',
            'Sinusal_150629',
            'Simulation_01_210209_001_001',
            'LA_RSPV_150113'],
            'test': ['Simulation_01_200428_001_007',
            'Simulation_01_200428_001_004',
            'Simulation_01_201223_001_002',
            'Simulation_01_200212_001_4',
            'Simulation_01_191001_001_005',
            'LA_RSPV_CAF_150115',
            'RA_RAA_141216',
            'LA_LSPV_150113',
            'Simulation_01_191001_001_007',
            'Simulation_01_200428_001_002',
            'Simulation_01_200212_001_6',
            'Simulation_01_200428_001_003']}}
        
        print("Cross Validation experiment. Loading precomputed fold ", fold)
        return precomputed_folds[f"fold_{fold}"]