
import pandas as pd
import numpy as np
from sklearn.utils import resample
from collections import Counter

from typing import List, Optional, Union, Set



class StratifiedSplit:
    def __init__(self,classes_to_oversample=[],
                discard_classes=[],
                oversampling=True,
                annotation_dir="/home/profes/miriamgf/tesis/Autoencoders/Data/annotations.csv", 
                val_source=None,
                test_source= None):

        self.annotation_dir = annotation_dir
        self.discard_classes = discard_classes
        self.oversampling=oversampling
        self.oversampling_rate=10
        self.classes_to_oversample=classes_to_oversample
        self.classes_to_strat=[0,1,2,3,4,5]
        self.val_source = val_source
        self.test_source = test_source
    
    def _read_names_source(self, src: Optional[Union[str, List[str]]]) -> Set[str]:
        """
        Lee una fuente de nombres (lista o path a txt). Devuelve set normalizado (strip).
        """
        if src is None:
            return set()
        if isinstance(src, list):
            return set([str(x).strip() for x in src])
        # si es string, asumimos path a txt con un nombre por línea
        with open(src, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        return set(names)


        
    def load_process_annotations(self, deterministic_group=None, select_classes=None):
        """
        Load annotation from a given path.
        """
        # Load
        df_annotation_complexity= pd.read_csv(self.annotation_dir, delimiter=';')

        #Discard patients from discard_classes
        if len(self.discard_classes)>0:
            if deterministic_group is not None:

                df_annotation_complexity['Simulation_Name'] = df_annotation_complexity['Simulation_Name'].str.strip()
                deterministic_group = [x.strip() for x in deterministic_group]

                # Filtrar el DataFrame para obtener solo clase 2 o 4
                names_to_exclude = df_annotation_complexity[df_annotation_complexity['Complexity'].isin(self.discard_classes)]['Simulation_Name'].tolist()

                # Filtrar la lista original
                filtered_group = [name for name in deterministic_group if name not in names_to_exclude]

                # Resultado
                return filtered_group


            df_annotation_complexity = df_annotation_complexity[~df_annotation_complexity['Complexity'].isin(self.discard_classes)]
        if select_classes is not None:
            df_annotation_complexity = df_annotation_complexity[df_annotation_complexity['Complexity'].isin(select_classes)]

                    # Select only patients with complexity in classes_to_strat
    
        return df_annotation_complexity

    
    def create_stratified_groups_with_fixed_val_test(
            self,
            df: pd.DataFrame,
            fixed_val: Set[str],
            fixed_test: Set[str],
            random_state: int = 42,
            train_frac: float = 0.75,
            val_frac: float = 0.10,
            test_frac: float = 0.15,
        ):
        """
        Usa val/test fijos y completa por clase hasta alcanzar proporciones objetivo.
        El resto va a TRAIN. Oversampling solo en TRAIN.
        """
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Las fracciones deben sumar 1."

        # Normalizar nombres presentes en df
        df = df.copy()
        df["Simulation_Name"] = df["Simulation_Name"].astype(str).str.strip()
        all_names = set(df["Simulation_Name"].tolist())

        # Filtrar fijos que no están en el CSV
        fixed_val = set([n for n in fixed_val if n in all_names])
        fixed_test = set([n for n in fixed_test if n in all_names])

        # Comprobar solapes
        overlap = fixed_val.intersection(fixed_test)
        if overlap:
            raise ValueError(f"Hay pacientes en VAL y TEST a la vez: {list(overlap)[:5]} ...")

        # DataFrames por split fijo
        df_val_fixed = df[df["Simulation_Name"].isin(fixed_val)].copy()
        df_test_fixed = df[df["Simulation_Name"].isin(fixed_test)].copy()
        df_pool = df[~df["Simulation_Name"].isin(fixed_val.union(fixed_test))].copy()

        def _print_counts(tag, ddf):
            cnt = ddf["Complexity"].value_counts().sort_index()
            print(f"Distribución {tag}:")
            if cnt.empty:
                print("  (vacío)")
            else:
                for k, v in cnt.items():
                    print(f"  Clase {k}: {v}")

        _print_counts("VAL (fijo)", df_val_fixed)
        _print_counts("TEST (fijo)", df_test_fixed)

        train_patients = []
        val_patients = df_val_fixed["Simulation_Name"].tolist()
        test_patients = df_test_fixed["Simulation_Name"].tolist()

        rng = np.random.default_rng(seed=random_state)

        for class_i in self.classes_to_strat:
            # Conjunto completo por clase
            df_c_all = df[df["Complexity"] == class_i].copy()
            n_total = len(df_c_all)
            if n_total == 0:
                print(f"Clase {class_i}: 0 pacientes -> se omite")
                continue

            # Objetivos por clase
            n_val_tgt = int(round(n_total * val_frac))
            n_test_tgt = int(round(n_total * test_frac))
            # train objetivo se deduce del remanente (puede quedar >/< del 70% si los fijos exceden)
            # pero no necesitamos fijarlo; saldrá de lo que quede.

            # Recuento fijos por clase
            n_val_fix = (df_val_fixed["Complexity"] == class_i).sum()
            n_test_fix = (df_test_fixed["Complexity"] == class_i).sum()

            # Pool disponible para completar en esta clase
            df_c_pool = df_pool[df_pool["Complexity"] == class_i].sample(frac=1, random_state=random_state).reset_index(drop=True)

            # Cuánto falta para llegar al objetivo
            need_val = max(0, n_val_tgt - n_val_fix)
            need_test = max(0, n_test_tgt - n_test_fix)

            # Si los fijos superan el target, avisamos y dejamos así
            if n_val_fix > n_val_tgt:
                print(f"[Aviso] Clase {class_i}: VAL fijo ({n_val_fix}) > target ({n_val_tgt}). Se respeta y no se completa.")
            if n_test_fix > n_test_tgt:
                print(f"[Aviso] Clase {class_i}: TEST fijo ({n_test_fix}) > target ({n_test_tgt}). Se respeta y no se completa.")

            # Completar VAL
            if need_val > 0:
                take = min(need_val, len(df_c_pool))
                chosen = df_c_pool.iloc[:take]["Simulation_Name"].tolist()
                val_patients.extend(chosen)
                df_c_pool = df_c_pool.iloc[take:]  # actualizar pool

            # Completar TEST
            if need_test > 0:
                take = min(need_test, len(df_c_pool))
                chosen = df_c_pool.iloc[:take]["Simulation_Name"].tolist()
                test_patients.extend(chosen)
                df_c_pool = df_c_pool.iloc[take:]  # actualizar pool

            # El resto (si queda) -> TRAIN
            train_subset = df_c_pool["Simulation_Name"].tolist()

            # Oversampling en TRAIN si aplica
            if (class_i in self.classes_to_oversample) and self.oversampling and len(train_subset) > 0:
                train_subset = resample(
                    train_subset,
                    replace=True,
                    n_samples=len(train_subset) * self.oversampling_rate,
                    random_state=random_state
                )

            train_patients.extend(train_subset)

            # Reporte por clase
            print(f"Clase {class_i} -> total:{n_total} | val_fix:{n_val_fix} test_fix:{n_test_fix} "
                f"| val_add:{max(0, min(need_val, (n_total - n_val_fix - n_test_fix)))} "
                f"| test_add:{max(0, min(need_test, (n_total - n_val_fix - n_test_fix - max(0, min(need_val, (n_total - n_val_fix - n_test_fix))))))} "
                f"| train_final(sin oversample): {len(set(train_subset))}")

        # Tamaños finales
        print("Tamaños finales:")
        print("  TRAIN:", len(train_patients))
        print("  VAL  :", len(val_patients))
        print("  TEST :", len(test_patients))

        # Comprobar proporciones de TRAIN (únicos, sin oversampling)
        sim_to_complexity = dict(zip(df["Simulation_Name"], df["Complexity"]))
        complexities_in_train = [sim_to_complexity[s] for s in set(train_patients) if s in sim_to_complexity]
        final_counts = Counter(complexities_in_train)
        for k in sorted(final_counts):
            print(f'Complejidad {k} en TRAIN (únicos, sin oversampling): {final_counts[k]}')

        return train_patients, test_patients, val_patients


    
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
    
    def get_test_for_inference(self, experiment_dir):
        """
        Get the test patients for inference from the original experiment dir.
        """
        path= experiment_dir + "/test_models.txt"
        with open(path, "r") as f:
            test_models_strat = [line.strip() for line in f if line.strip()]

        return test_models_strat
    



    def __call__(self, *args, **kwds):
        
        df_annotation_complexity_selected=self.load_process_annotations()

        # 2) Cargar listas fijas (si existen)
        fixed_val = self._read_names_source(self.val_source)
        fixed_test = self._read_names_source(self.test_source)

        if fixed_val or fixed_test:
            # usar modo con val/test fijos
            train_patients, test_patients, val_patients = self.create_stratified_groups_with_fixed_val_test(
                df_annotation_complexity_selected, fixed_val, fixed_test
            )
        else:

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
    stratified_split = StratifiedSplit(classes_to_oversample=[4],
                                       discard_classes =[0,1, 3, 5],
                                       oversampling=True, 
                                        val_source=None,
                                        test_source=["Simulation_01_200316_001_  5",
                                                "Simulation_01_200212_001_  7",
                                                'Simulation_01_210205_001_003',
                                                'Simulation_01_200428_001_009',
                                                'Simulation_01_210209_001_003'])
    df_annotation_complexity_selected=stratified_split()
