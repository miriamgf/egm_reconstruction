import numpy as np
import tensorflow as tf

class ConcatSynthetic:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.offset_pid_ = 0
        self.patient_order_ = None
        self.patient_order_real_ = None
        self.patient_order_synt_ = None

    @staticmethod
    def _to_sample_ids(ids_vector, n_samples, T):
        ids_2d = np.asarray(ids_vector).reshape(n_samples, T)
        return ids_2d[:, 0]

    def _build_patient_groups(self, patient_ids, shuffle_patients=True, shuffle_within=False):
        groups = {}
        for i, pid in enumerate(patient_ids):
            pid = int(pid)
            groups.setdefault(pid, []).append(i)
        pids = list(groups.keys())
        if shuffle_patients:
            self.rng.shuffle(pids)
        if shuffle_within:
            for pid in pids:
                self.rng.shuffle(groups[pid])
        return [groups[pid] for pid in pids], pids

    @staticmethod
    def _round_robin(groups_a, groups_b):
        i = j = 0
        merged, src_flags = [], []
        while i < len(groups_a) or j < len(groups_b):
            if i < len(groups_a):
                merged.append(groups_a[i]); src_flags.append('r'); i += 1
            if j < len(groups_b):
                merged.append(groups_b[j]); src_flags.append('s'); j += 1
        return merged, src_flags

    @staticmethod
    def _concat_and_reorder(real_arr, synt_arr, perm=None, is_tf=True):
        if is_tf:
            combined = tf.concat([real_arr, synt_arr], axis=0)
            return tf.gather(combined, perm, axis=0) if perm is not None else combined
        else:
            combined = np.concatenate([real_arr, synt_arr], axis=0)
            return combined[perm] if perm is not None else combined

    @staticmethod
    def _safe_get(lst, idx, fallback):
        try:
            return lst[int(idx)]
        except Exception:
            return fallback

    def fit_transform(
        self,
        x_train, y_train, BSPM_train,
        AF_models_train, class_complexity_list_train,
        synt_x_train, synt_y_train, synt_BSPM_train,
        synt_AF_models_train, synt_class_complexity_list_train,
        *,
        concat: str = "alternan",          # "alternan" o "final"
        reindex_synthetics: bool = True,   # suma offset a sintéticos para evitar colisión
        shuffle_patients: bool = True,
        shuffle_within_patient: bool = False,
        balance_min: bool = False,
        tf_tensors=('x_train','y_train','BSPM_train'),
        all_model_names: list | None = None,
        synt_all_model_names: list | None = None
    ):
        # IDs por muestra
        n_r, T_r = x_train.shape[0], x_train.shape[1]
        n_s, T_s = synt_x_train.shape[0], synt_x_train.shape[1]
        AF_r = self._to_sample_ids(AF_models_train, n_r, T_r)
        AF_s = self._to_sample_ids(synt_AF_models_train, n_s, T_s)

        # Reindexar sintéticos si procede
        if reindex_synthetics:
            self.offset_pid_ = int(np.max(AF_r)) + 1
            AF_s = AF_s + self.offset_pid_
        else:
            self.offset_pid_ = 0

        patient_order_names = None

        if concat == "alternan":
            # Agrupar y alternar por paciente
            real_groups, _ = self._build_patient_groups(AF_r, shuffle_patients, shuffle_within_patient)
            synt_groups, _ = self._build_patient_groups(AF_s, shuffle_patients, shuffle_within_patient)
            if balance_min:
                k = min(len(real_groups), len(synt_groups))
                real_groups, synt_groups = real_groups[:k], synt_groups[:k]

            merged_groups, src_flags = self._round_robin(real_groups, synt_groups)

            # Permutación y orden de pacientes
            perm = []
            seen_pid = set()
            pid_seq_once = []
            for src, group in zip(src_flags, merged_groups):
                if src == 'r':
                    perm.extend(group)
                    pid = AF_r[group[0]]
                else:
                    perm.extend([n_r + idx for idx in group])
                    pid = AF_s[group[0]]
                if pid not in seen_pid:
                    pid_seq_once.append(pid); seen_pid.add(pid)

            perm = np.asarray(perm, dtype=np.int64)
            self.patient_order_ = np.array(pid_seq_once)
            self.patient_order_real_ = np.array([p for p in self.patient_order_ if p < self.offset_pid_])
            self.patient_order_synt_ = np.array([p for p in self.patient_order_ if p >= self.offset_pid_])


            # Concat + reorder
            def is_tf_tensor(name): return name in tf_tensors
            x_train  = self._concat_and_reorder(x_train,  synt_x_train,  perm, is_tf=is_tf_tensor('x_train'))
            y_train  = self._concat_and_reorder(y_train,  synt_y_train,  perm, is_tf=is_tf_tensor('y_train'))
            BSPM_train = self._concat_and_reorder(BSPM_train, synt_BSPM_train, perm, is_tf=is_tf_tensor('BSPM_train'))
            AF_train = self._concat_and_reorder(AF_r, AF_s, perm, is_tf=False)
            class_complexity_train = self._concat_and_reorder(
                np.asarray(class_complexity_list_train),
                np.asarray(synt_class_complexity_list_train),
                perm, is_tf=False
            )
        else:
            # Append clásico
            perm = None
            self.patient_order_ = None
            self.patient_order_real_ = None
            self.patient_order_synt_ = None
            patient_order_names = None

            def is_tf_tensor(name): return name in tf_tensors
            x_train  = self._concat_and_reorder(x_train,  synt_x_train,  perm, is_tf=is_tf_tensor('x_train'))
            y_train  = self._concat_and_reorder(y_train,  synt_y_train,  perm, is_tf=is_tf_tensor('y_train'))
            BSPM_train = self._concat_and_reorder(BSPM_train, synt_BSPM_train, perm, is_tf=is_tf_tensor('BSPM_train'))
            AF_train = self._concat_and_reorder(AF_r, AF_s, perm, is_tf=False)
            class_complexity_train = self._concat_and_reorder(
                np.asarray(class_complexity_list_train),
                np.asarray(synt_class_complexity_list_train),
                perm, is_tf=False
            )

        return {
            'x_train': x_train,
            'y_train': y_train,
            'BSPM_train': BSPM_train,
            'AF_models_train': AF_train,
            'class_complexity_list_train': class_complexity_train,
            'patient_order': self.patient_order_,
            'patient_order_real': self.patient_order_real_,
            'patient_order_synt': self.patient_order_synt_,
            'patient_order_names': patient_order_names,   # ✅ nombres reales + sintéticos
        }
