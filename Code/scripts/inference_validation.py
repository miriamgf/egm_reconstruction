import sys
sys.path.append('../Code')
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools_.tools_inference import *
from tools_.tools import corr_pearson_cols
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scripts.evaluate_function import *

batch_size = 400

egm = load_EGM(SR=True, 
            subsampling=True, 
            fs_sub=100, 
            n_nodes_regression=683
            )

#Interpolate

bsps = load_BSPS(subsampling=True, 
              fs_sub=100, 
              batch_size=batch_size
            )
Y_model = np.zeros(len(bsps))
bsps = normalize_by_models(bsps, Y_model)
center_function = lambda x: x - x.mean(axis=0)
bsps = center_function(bsps)

plt.figure(figsize=(20, 7))
plt.subplot(2, 1, 1)
plt.plot(egm[0:1000, 0])
plt.title('EGM')
plt.subplot(2, 1, 2)
plt.plot(bsps[0:1000, 0, 0])
plt.title('BSPS')
plt.savefig('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/Inference/egm.png')
plt.show()

plt.figure()
plt.imshow(egm)
plt.savefig('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/Inference/egm_interpolated.png')
plt.show()

plt.figure()
plt.imshow(bsps[0, :, :])
plt.savefig('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/Inference/bsps.png')
plt.show()

# Batch generation

bsps_batches = reshape(
            bsps,
            (
                int(len(bsps) / batch_size),
                batch_size,
                bsps.shape[1],
                bsps.shape[2],
                1,
            ),
        )
egm_batches = reshape(
            egm,
            (
                int(len(egm) / batch_size),
                batch_size,
                egm.shape[1],
            ),
        )

#Load model
model_path = '/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_CINC/20240827-112359_EXP_0/model_mo.h5'
model = load_model(model_path)
#Inference
prediction  = model.predict(bsps_batches, batch_size=1) # x_test=[#batches, batch_size, 12, 32, 1]
prediction = prediction[1]
prediction_flat = prediction.reshape((prediction.shape[0]*prediction.shape[1], prediction.shape[2]))
egm_flat = egm_batches.reshape((prediction.shape[0]*prediction.shape[1], prediction.shape[2]))

prediction = normalize_by_models(prediction_flat, Y_model)

for node in range(0, 100):

    plt.figure(figsize=(20, 7))
    plt.plot(prediction[0:500, node], label = 'prediction')
    plt.plot(egm_flat[0:500, node], label ='real')
    plt.legend()
    plt.savefig('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/Validation_real//Prediction'+str(node)+'.png')

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(prediction[0:500, :], label = 'prediction', cmap = 'grey')
plt.title('prediction')
plt.subplot(1, 2, 2)

plt.imshow(egm_flat[0:500, :], label ='real', cmap = 'grey')
plt.title('real')
plt.legend()
plt.savefig('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/Validation_real/Prediction_2d.png')



(global_loss_test,
mse_test_ae,
mse_test_regressor,
mae_test_ae,
mae_test_regressor) = model.evaluate(bsps_batches, [bsps_batches, egm_batches], batch_size=1)

print('MSE: ', mse_test_regressor)



corr = corr_pearson_cols(prediction, egm_flat)
plt.figure()
plt.boxplot(corr)
plt.savefig('/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/figures/Validation_real/corr.png')

