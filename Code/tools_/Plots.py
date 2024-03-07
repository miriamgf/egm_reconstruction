import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools import *
from generators import *
from scipy import *




def define_image(time_instant, Tensor):
    image=Tensor[time_instant,: ,:]
    return image
def define_frames(Tensor):
    ims = []
    for i in range(200):
        ims.append([Tensor[i, :, :]])
    return ims

def plotting_before_AE(X_1channel, x_train, x_test, x_val, Y_model, AF_models ):

    # Signal plot 
    plt.figure(figsize=(10, 3))
    a=randint(0, 5)
    b=randint(0, 15)
    s=randint(0, len(X_1channel)-501 )
    title="500 samples from BSP from nodes {} x {} in the model {}".format(a,b,Y_model[s] )
    #plt.plot(X_1channel[s:s+200, a , b])
    plt.plot(X_1channel[0:1000, 0 , 0])
    plt.ylabel('Amplitude')
    plt.xlabel('Samples')
    plt.title(title)
    plt.show()

    # %% [markdown]
    # ## 1.2 Subplot of 5 frames (images) in the range of samples r1:r2

    # %%
    r1=1
    r2=10

    model_index = np.where((Y_model==180))

    plt.figure(figsize=(10, 20), layout='tight')
    for i in range(r1,r2):
        image=define_image(i, X_1channel)
        plt.subplot(r2-r1, 1, i)
        plt.imshow(image) #map='jet')
        title="Image corresponding to sample {}".format(i)
        plt.title(title)
        
    plt.show()




    # %% [markdown]
    # ## 1.3 PLOT INTERPOLATED IMAGES 




    # %%
    #Plot
    x_train_or, x_test_or, x_val_or=x_train, x_test, x_val

    F= 1
    plt.figure()
    plt.subplot(4,1,1)
    image=define_image(F,x_train_or)
    plt.imshow(image, cmap='jet')
    plt.title('Interpolated, sample s={}'. format(F))
    plt.colorbar()

    F= 100
    plt.subplot(4,1,2)
    image=define_image(F,x_train )
    plt.imshow(image, cmap='jet')
    plt.title('Interpolated, sample s={}'. format(F))
    plt.colorbar()

    F= 200
    plt.subplot(4,1,3)
    image=define_image(F,x_train )
    plt.imshow(image, cmap='jet')
    plt.title('Interpolated, sample s={}'. format(F))
    plt.colorbar()
    plt.suptitle('2x2 Plots'  )

    F= 300
    plt.subplot(4,1,4)
    image=define_image(F,x_train )
    plt.imshow(image, cmap='jet')
    plt.title('Interpolated, sample s={}'. format(F))
    plt.colorbar()
    plt.suptitle('2x2 Plots'  )


    plt.show()

def plot_Welch_periodogram(x_test, latent_vector, decoded_imgs, fs= 50):
    input_signal= x_test
    LS_signal= latent_vector
    output_signal= decoded_imgs
    nperseg_value=200

    fig=plt.figure(layout='tight')

    plt.subplot(3,1,1)

    #Input
    for height in range(0,10, 2):
        for width in range(0,10, 2):
        
            f, Pxx_den = signal.welch(input_signal[:, height,width], fs, nperseg=len(input_signal[:, height,width]),
                                    noverlap = nperseg_value // 2, scaling = 'density', detrend='linear')
            plt.plot(f, Pxx_den,linewidth=0.5)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('PSD')
            titlee="PSD of input signal. nodes ({} , {})".format(height,width )
            plt.title('Input signal')
    plt.show()

    # Output
    plt.subplot(3,1,2)
    for height in range(0,10, 2):
        for width in range(0,10, 2):
        
            f, Pxx_den = signal.welch(output_signal[:, height,width],fs, nperseg=len(output_signal[:, height,width]),
                                    noverlap = nperseg_value // 2, scaling = 'density', detrend='linear' )
            plt.plot(f, Pxx_den,linewidth=0.5)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('PSD')
            titlee="PSD of output signal. nodes ({} , {})".format(height,width )
            plt.title('Output signal')
    plt.show()

    # Latent Space
    plt.subplot(3,1,3)
    for height in range(0,3):
        for width in range(0,4):
            for filters in range(0,12,2):
                
                length= LS_signal[:, height,width, filters]
        
                f, Pxx_den = signal.welch(LS_signal[:, height,width, filters], fs, nperseg=len(LS_signal[:, height,width, filters]),
                                    noverlap = nperseg_value // 2, scaling = 'density', detrend='linear' )
                plt.plot(f, Pxx_den, linewidth=0.5)
                plt.xlabel('frequency [Hz]')
                plt.ylabel('PSD')
                titlee="PSD of Latent Space signal. nodes ( - )".format(height,width )
                plt.title('latent Space')

    plt.savefig('Figures/Periodogram_AE.png')
    plt.show()
    fig.suptitle("PSD", fontsize=15)



def plot_Welch_reconstruction(latent_vector_test, estimate_egms_n, fs, n_nodes,y_test_subsample,  nperseg_value=200):

    LS_signal= latent_vector_test
    output_signal= estimate_egms_n

    fig=plt.figure(layout='tight', figsize=(10, 6))

    plt.subplot(3,1,1)

    #reconstruction
    for height in range(0,n_nodes-1, 5):
        
        f, Pxx_den = scipy.signal.welch(output_signal[:, height], fs, nperseg=nperseg_value, noverlap = None, scaling = 'density', detrend='linear')
        plt.plot(f, Pxx_den,linewidth=0.5)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        #plt.ylim([0,0.2])

        titlee="PSD Welch of EGM signal estimation. node {}".format(height )
        plt.title('EGM signals reconstruction')


    # Latent Space
    plt.subplot(3,1,2)
    for height in range(0,3):
        for width in range(0,4):
            for filters in range(0,12,2):
        
                f, Pxx_den = scipy.signal.welch(LS_signal[:, height,width, filters], fs, nperseg=nperseg_value,noverlap = None, scaling = 'density', detrend='linear')
                plt.plot(f, Pxx_den,linewidth=0.5)
                plt.xlabel('frequency [Hz]')
                plt.ylabel('PSD [V**2/Hz]')
                titlee="PSD Welch of Latent Space signal. nodes ( - )".format(height,width )
                plt.title('latent Space')
                #plt.ylim([0,0.2])

    fig.suptitle("Welch Periodogram (window size=200 samples)", fontsize=15)

    #Input
    plt.subplot(3,1,3)

    for height in range(0,n_nodes-1, 5):
        
        f, Pxx_den = scipy.signal.welch(y_test_subsample[:, height], fs, nperseg=nperseg_value,
                                    noverlap = None, scaling = 'density', detrend='linear')
        plt.plot(f, Pxx_den,linewidth=0.5)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        titlee="PSD Welch of EGM signal estimation. node {}".format(height )
        plt.title('EGM signals Real')
        #plt.ylim([0,0.2])
    plt.savefig('Figures/Periodogram_rec.png')
    plt.show()



    