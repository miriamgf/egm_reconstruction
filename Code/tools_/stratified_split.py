
import pandas as pd
import numpy as np
from sklearn.utils import resample
from collections import Counter


class StratifiedSplit:
    def __init__(self,classes_to_oversample, discard_classes=[], oversampling=True):

        self.annotation_dir = "/home/profes/miriamgf/tesis/Autoencoders/Data/annotations.csv"
        self.discard_classes = discard_classes
        self.oversampling=oversampling
        self.oversampling_rate=10
        self.classes_to_oversample=classes_to_oversample
        self.classes_to_strat=[0,1,2,3,4,5]

    def load_process_annotations(self, select_classes=None):
        """
        Load annotation from a given path.
        """
        # Load
        df_annotation_complexity= pd.read_csv(self.annotation_dir, delimiter=';')

        #Discard patients from discard_classes
        if len(self.discard_classes)>0:
            df_annotation_complexity = df_annotation_complexity[~df_annotation_complexity['Complexity'].isin(self.discard_classes)]
        if select_classes is not None:
            df_annotation_complexity = df_annotation_complexity[df_annotation_complexity['Complexity'].isin(select_classes)]

                    # Select only patients with complexity in classes_to_strat
    
        return df_annotation_complexity
    
    def create_stratified_groups(self, df_annotation_complexity_selected):

        train_patients= []
        test_patients= []
        val_patients= []

        for class_i in self.classes_to_strat:
            df_class_i = df_annotation_complexity_selected[df_annotation_complexity_selected["Complexity"] == class_i]

            print('class', class_i, " has ", len(df_class_i), " patients")

            # Mezclar aleatoriamente
            df_class_i = df_class_i.sample(frac=1, random_state=42).reset_index(drop=True)

            n_total = len(df_class_i)
            n_train = int(n_total * 0.7)
            n_val = int(n_total * 0.15)
            n_test = n_total - n_train - n_val  

            # Dividir
            df_class_i_train = df_class_i.iloc[:n_train]
            df_class_i_val = df_class_i.iloc[n_train:n_train + n_val]
            df_class_i_test = df_class_i.iloc[n_train + n_val:]

            train_subset= np.array(df_class_i_train.Simulation_Name)
            test_subset= np.array(df_class_i_test.Simulation_Name)
            val_subset= np.array(df_class_i_val.Simulation_Name)

            print("train_subset", len(train_subset))
            print("test_subset", len(test_subset))
            print("val_subset", len(val_subset))

            #Apply oversampling only to training subset
            if class_i in self.classes_to_oversample and self.oversampling:
                # Oversample the training set
                train_subset = resample(train_subset, 
                                        replace=True,     # sample with replacement
                                        n_samples=len(train_subset) * self.oversampling_rate,    # to match majority class
                                        random_state=42)


            train_patients.extend(train_subset)
            test_patients.extend(test_subset)
            val_patients.extend(val_subset)
        
        print("train_patients: ", train_patients)
        print("test_patients: ", test_patients)
        print("val_patients: ", val_patients)

        return train_patients, test_patients, val_patients
    



    def __call__(self, *args, **kwds):
        
        df_annotation_complexity_selected=self.load_process_annotations()
        train_patients, test_patients, val_patients=self.create_stratified_groups(df_annotation_complexity_selected)

        #Check final proportions
        sim_to_complexity = dict(zip(df_annotation_complexity_selected['Simulation_Name'], 
                                    df_annotation_complexity_selected['Complexity']))
        complexities_in_train = [sim_to_complexity[sim.strip()] for sim in train_patients if sim.strip() in sim_to_complexity]
        final_counts = Counter(complexities_in_train)
        for k in sorted(final_counts):
            print(f'Complejidad {k}: {final_counts[k]} veces')

        return train_patients, test_patients, val_patients
        

if __name__ == "__main__":
    stratified_split = StratifiedSplit(classes_to_oversample=1, oversampling=True)
    df_annotation_complexity_selected=stratified_split()
