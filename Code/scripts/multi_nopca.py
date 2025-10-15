import import_data
import numpy as np

egms_all_models = import_data.egms_all_models
egms_values = import_data.egms_values
egms_filtered_norm = import_data.egms_filtered_norm
bsps_all_models = import_data.bsps_all_models
bsps_64_all_models = import_data.bsps_64_all_models
transfer_matrices = import_data.transfer_matrices
models_names = import_data.models_names
bsps_64_all_models_names = import_data.bsps_64_all_models_names

print(bsps_64_all_models_names[:100])
print(len(bsps_64_all_models_names))
# models_name_full = []
# for name in models_names:
#     models_name_full.extend([name] * 10)

models_name_full = bsps_64_all_models_names
# Verifica que coinciden
print(len(models_name_full))  # Debería dar 240
filtered_models = []
filtered_models_names = []

for model, name_model in zip(bsps_64_all_models, models_name_full):
    new_model = []
    new_model2 = []

    for node in model:
        original_length = len(node)

        if original_length < 2000:
            continue

        if 2000 <= original_length <= 2501:
            node = node[:2000]
            new_model.append(node)
        elif 4000 <= original_length <= 4501:
            node = node[:4000]
            new_model.append(node[0:2000])
            new_model2.append(node[2000:4000])
        else:
            continue


    if len(new_model) > 0:
        filtered_models.append(new_model)
        filtered_models_names.append(name_model)  # Añadimos el nombre del modelo
    if len(new_model2) > 0:
        filtered_models.append(new_model2)
        filtered_models_names.append(name_model)
        
bsps_64_filtered_models = filtered_models

print(len(bsps_64_filtered_models))
print(len(filtered_models_names))
print(filtered_models_names)
print(filtered_models_names[140])

from collections import OrderedDict
unique_models = list(OrderedDict.fromkeys(filtered_models_names))
print(len(unique_models))
print(unique_models)

model_to_indices = {m: [] for m in unique_models}  # DICCIONARIO CON NOMBRES DE MODELOS Y SUS INDICES EN filtered_models_names
for i, name in enumerate(filtered_models_names):
    if name in unique_models:
        model_to_indices[name].append(i)
print(f"Índices de modelos: {model_to_indices}")


import pandas as pd
from sklearn.model_selection import train_test_split


# Leer el CSV
csv_df = pd.read_csv('annotations.csv', header=None)

# Separar por ';'
csv_split = csv_df[0].str.split(';', expand=True)
csv_names = csv_split[0].str.strip().tolist()  # Eliminar espacios
csv_labels = csv_split[1].astype(int).tolist()

# Lista de modelos únicos que quieres filtrar (asegúrate de que está definida)
# unique_models = [...]

# Crear un diccionario con todos los nombres y etiquetas
name_to_label = dict(zip(csv_names, csv_labels))

# Filtrar sólo los que están en unique_models
filtered_unique_models = [m for m in unique_models if name_to_label.get(m) != 1]
print(len(filtered_unique_models))

filtered_labels = [name_to_label[m] for m in filtered_unique_models]

train_models, test_models, train_labels, test_labels = train_test_split(
    filtered_unique_models,
    filtered_labels,
    test_size=0.3,
    stratify=filtered_labels,
    random_state=42
)

# Mostrar resultados
print(filtered_labels)
print(len(filtered_labels))

print("Total modelos únicos con etiqueta:", len(filtered_unique_models))


print("Modelos de entrenamiento:", len(train_models))
print("Modelos de prueba:", len(test_models))
print("Etiquetas de entrenamiento:", len(train_labels))
print("Etiquetas de prueba:", len(test_labels))

# Opcional: Ver distribución
from collections import Counter
print("Distribución train:", Counter(train_labels))
print("Distribución test:", Counter(test_labels))


X_train = []
y_train = []

for model_name, label in zip(train_models, train_labels):
    indices = model_to_indices.get(model_name, [])
    for idx in indices:
        X_train.append(filtered_models[idx])
        y_train.append(label)

X_test = []
y_test = []

for model_name, label in zip(test_models, test_labels):
    indices = model_to_indices.get(model_name, [])
    for idx in indices:
        X_test.append(filtered_models[idx])
        y_test.append(label)

# Verificación rápida
print(f"Total X_train: {len(X_train)}, y_train: {len(y_train)}")
print(f"Total X_test: {len(X_test)}, y_test: {len(y_test)}")

print(Counter(y_train))
print(Counter(y_test))

X_train = np.array(X_train)
X_test = np.array(X_test)


# WAVELET SCATTERING TRANSFORM

from kymatio import Scattering1D

def apply_wavelet_scattering(X, J=4, Q=(16, 16)):
    # APLICAR SCATTERING A CADA MODELO
    J = 4
    Q = (16,16)

    ws_coeffs = []  

    for model in X:
        model_coeffs = []  
        
        for j in range(len(model)):
            signal = np.array(model[j])  
            len_signal = len(signal)
            
            scattering = Scattering1D(J=J, shape=(len_signal,), Q=Q)
            Sx = scattering(signal)  
            model_coeffs.append(Sx)  
        ws_coeffs.append(np.stack(model_coeffs))
    return np.array(ws_coeffs)


#original_data NO PCA
X_train_ws_no_pca = apply_wavelet_scattering(X_train)

X_train_no_pca_final = X_train_ws_no_pca.reshape(X_train_ws_no_pca.shape[0], -1)
print(f"Scattering aplicado por modelo. Ejemplo dimensiones: {X_train_ws_no_pca[0].shape} ")


X_test_ws_no_pca = apply_wavelet_scattering(X_test)


print(f"Scattering aplicado por modelo. Ejemplo dimensiones: {X_train_ws_no_pca[0].shape} ")

X_test_no_pca_final = X_test_ws_no_pca.reshape(X_test_ws_no_pca.shape[0], -1)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, log_loss, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import recall_score, precision_score
def random_forest_100(X_train_final, y_train, X_test_final, y_test):
    # Entrenar el modelo Random Forest
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=None, bootstrap=True, random_state=42, n_jobs=-1)
    rf.fit(X_train_final, y_train)

    # Predecir clases
    y_pred = rf.predict(X_test_final)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Obtener probabilidades predichas (necesarias para AUC multiclase y log loss)
    y_proba = rf.predict_proba(X_test_final)  # shape (n_samples, n_classes)

    # AUC ROC multiclase requiere que las etiquetas estén binarizadas (one-hot)
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)

    # Calcular AUC-ROC multiclase (One-vs-Rest)
    auc_roc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr')
    print("AUC-ROC (multiclase OVR):", auc_roc)

    # Calcular Log Loss
    loss = log_loss(y_test_bin, y_proba)
    print("Log loss:", loss)
    return rf


rf_no_pca = random_forest_100(X_train_no_pca_final, y_train, X_test_no_pca_final, y_test)



from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
# Distribución de búsqueda
def tunning_random_forest(X_train_final, y_train, rf):
    # Crear el modelo Random Forest
    param_dist = {
    'max_depth': [None] + list(range(10, 40, 10)),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None]
}
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                   n_iter=50, cv=5, random_state=42, n_jobs=1,
                                   verbose=2, scoring='accuracy')

    
    random_search.fit(X_train_final, y_train)
    best_model = random_search.best_estimator_
    print("Mejores hiperparámetros:", random_search.best_params_)
    

tunning_random_forest(X_train_no_pca_final, y_train, rf_no_pca)
