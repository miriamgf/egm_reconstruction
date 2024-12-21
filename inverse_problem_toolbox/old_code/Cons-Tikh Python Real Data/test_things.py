# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:04:48 2020

@author: Miguel Ángel
"""
t=np.linspace(0,x_hat_tikh0.shape[1]/fs,x_hat_tikh0.shape[1])
plt.figure(),
plt.subplot(211),
plt.plot(t,x_hat_tikh0[472,:]/np.max(x_hat_tikh0[472,:])),

plt.xlim((0,2)),
plt.subplot(212),
plt.plot(f_tikh0,pxx_tikh0[472,:]),plt.xlim((0,50))


t=np.linspace(0,x_hat_tikh0.shape[1]/fs,x_hat_tikh0.shape[1])
plt.figure(),
plt.subplot(211),
plt.plot(t,x_hat_tikh0[472,:]/np.max(x_hat_tikh0[472,:])),

plt.xlim((0,2)),
plt.subplot(212),
plt.plot(f,p),plt.xlim((0,50))

f, pxx = sigproc.welch(signals[:,100:-500],fs,nfft=2**13,nperseg=512,axis=1)


plt.close('all')
t=np.linspace(0,x_hat_tikh0.shape[1]/fs,x_hat_tikh0.shape[1])
for i in range(0,73):
    plt.figure(),
    plt.subplot(211),plt.title('Node: %d. DF= %.3f' %(known_nodes[i],df_cons1[known_nodes[i]]))
    plt.plot(t,x_ref_cons_full[known_nodes[i],:]/np.max(x_ref_cons_full[known_nodes[i],:]),'k'),
    plt.plot(t,x_hat_cons1[known_nodes[i],:]/np.max(x_hat_cons1[known_nodes[i],:]),'r'),plt.grid('on')
    plt.legend(('GT','Cons1'))
    plt.subplot(212)
    #f_gt,p_gt = scipy.signal.welch(x_ref_cons_full[known_nodes[i],100:-500],fs=fs,nperseg=2048,noverlap=None,nfft=2**13)
    #f_tikh0,p_tikh0 = scipy.signal.welch(x_hat_tikh0[known_nodes[i],100:-500],fs=fs,nperseg=2048,noverlap=None,nfft=2**13)
    plt.plot(f,pxx_ref[known_nodes[i],:]/np.max(pxx_ref[known_nodes[i],:]),'k'),
    plt.plot(f_cons1,pxx_cons1[known_nodes[i],:]/np.max(pxx_cons1[known_nodes[i],:]),'r'),plt.xlim((0,30)),plt.grid('on'),plt.legend(('GT','Cons1'))
    
    
    
plt.figure(),
plt.subplot(211),
t=np.linspace(0,x_hat_tikh0.shape[1]/fs,x_hat_tikh0.shape[1])
plt.plot(t,x_ref_cons_full[known_nodes[13],:]/np.max(x_ref_cons_full[known_nodes[13],:])),
plt.plot(t,x_hat_tikh0[known_nodes[13],:]/np.max(x_hat_tikh0[known_nodes[13],:])),
plt.plot(t,x_hat_cons1[known_nodes[13],:]/np.max(x_hat_cons1[known_nodes[13],:])),
plt.legend(('GT','Tikh0','Cons1')),plt.grid('on')

plt.subplot(212),
plt.plot(f,pxx_ref[known_nodes[13],:]/np.max(pxx_ref[known_nodes[13],:])),
plt.plot(f_tikh0,pxx_tikh0[known_nodes[13],:]/np.max(pxx_tikh0[known_nodes[13],:])),
plt.plot(f_cons1,pxx_cons1[known_nodes[13],:]/np.max(pxx_cons1[known_nodes[13],:]))
plt.legend(('GT','Tikh0','Cons1')),plt.grid('on'),plt.xlim((0,30)),


plt.close('all')
id_to_look = 697
plt.figure(),
plt.subplot(211),plt.title('Node: %d. DF=%.3f'%(id_to_look,df_cons1[id_to_look]))
t=np.linspace(0,x_hat_tikh0.shape[1]/fs,x_hat_tikh0.shape[1])
plt.plot(t,x_hat_tikh0[id_to_look,:]/np.max(x_hat_tikh0[id_to_look,:])),
plt.plot(t,x_hat_cons1[id_to_look,:]/np.max(x_hat_cons1[id_to_look,:])),
plt.legend(('Tikh0','Cons1')),
plt.grid('on')

plt.subplot(212),
plt.plot(f,pxx_tikh0[id_to_look,:]/np.max(pxx_tikh0[id_to_look,:])),
plt.plot(f_cons1,pxx_cons1[id_to_look,:]/np.max(pxx_cons1[id_to_look,:]))
plt.legend(('Tikh0','Cons1')),
plt.grid('on'),plt.xlim((0,30))


plt.close('all')
id_to_look = 697
plt.figure(),
plt.subplot(211),plt.title('Node: %d. DF=%.3f'%(id_to_look,df_tikh0[id_to_look]))
t=np.linspace(0,x_hat_tikh0.shape[1]/fs,x_hat_tikh0.shape[1])
plt.plot(t,signals_tikh0_df[id_to_look,:]/np.max(signals_tikh0_df[id_to_look,:])),
plt.plot(t,signals_cons1_df[id_to_look,:]/np.max(signals_cons1_df[id_to_look,:])),
plt.legend(('Tikh0','Cons1')),
plt.grid('on')

plt.subplot(212),
plt.plot(f_tikh0,pxx_tikh0[id_to_look,:]/np.max(pxx_tikh0[id_to_look,:])),
plt.plot(f_cons1,pxx_cons1[id_to_look,:]/np.max(pxx_cons1[id_to_look,:]))
plt.legend(('Tikh0','Cons1')),
plt.grid('on'),plt.xlim((0,30))


""" Phase analysis (known_nodes)"""
plt.close('all')
t=np.linspace(0,x_hat_tikh0.shape[1]/fs,x_hat_tikh0.shape[1])
for i in range(0,len(known_nodes_full)):
    signal_gt=x_ref_cons_full[known_nodes[i],:]/np.max(x_ref_cons_full[known_nodes[i],:])
    signal_tikh0=x_hat_tikh0[known_nodes[i],:]/np.max(x_hat_tikh0[known_nodes[i],:])
    signal_cons1=x_hat_cons1[known_nodes[i],:]/np.max(x_hat_cons1[known_nodes[i],:])

    instant_phase_gt = -np.arctan2(np.imag(sigproc.hilbert(signal_gt)),signal_gt)
    instant_phase_tikh0 = -np.arctan2(np.imag(sigproc.hilbert(signal_tikh0)),signal_tikh0)
    instant_phase_cons1 = -np.arctan2(np.imag(sigproc.hilbert(signal_cons1)),signal_cons1)

    pxx_gt_norm = pxx_ref[known_nodes[i],:]/np.max(pxx_ref[known_nodes[i],:])
    pxx_tikh0_norm = pxx_tikh0[known_nodes[i],:]/np.max(pxx_tikh0[known_nodes[i],:])
    pxx_cons1_norm = pxx_cons1[known_nodes[i],:]/np.max(pxx_cons1[known_nodes[i],:])

    plt.figure(),
    plt.subplot(231),plt.title('Signals. Node: %d.' %(known_nodes[i]))
    plt.plot(t,signal_gt,'b'),
    plt.plot(t,signal_tikh0,'r'),plt.grid('on'),plt.xlim((1,3))
    plt.legend(('GT','Tikh0'))
    plt.subplot(232),
    plt.title('Phase domain')
    plt.plot(t,instant_phase_gt,'b'),
    plt.plot(t,instant_phase_tikh0,'r'),plt.grid('on'),plt.xlim((1,3))
    plt.legend(('GT','Tikh0'))
    plt.subplot(233),    
    plt.plot(f,pxx_gt_norm,'b'),
    plt.plot(f_tikh0,pxx_tikh0_norm,'r'),plt.xlim((0,30)),plt.grid('on'),
    plt.title('DF_GT= %.3f Hz. DF_Tikh0=%.3f Hz.' %(df_ref[known_nodes[i]],df_tikh0[known_nodes[i]]))
    plt.legend(('GT','Tikh0'))
    
    plt.subplot(234),plt.title('Signals. Node: %d.' %(known_nodes[i]))
    plt.plot(t,signal_gt,'b'),
    plt.plot(t,signal_cons1,'r'),plt.grid('on'),plt.xlim((1,3))
    plt.legend(('GT','Cons1'))
    plt.subplot(235),
    plt.title('Phase domain')
    plt.plot(t,instant_phase_gt,'b'),
    plt.plot(t,instant_phase_cons1,'r'),plt.grid('on'),plt.xlim((1,3))
    plt.legend(('GT','Cons1'))
    plt.subplot(236),    
    plt.plot(f,pxx_gt_norm,'b'),
    plt.plot(f_cons1,pxx_cons1_norm,'r'),plt.xlim((0,30)),plt.grid('on'),
    plt.title('DF_GT= %.3f Hz. DF_Cons1=%.3f Hz.' %(df_ref[known_nodes[i]],df_cons1[known_nodes[i]]))
    plt.legend(('GT','Cons1'))
    
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    
    
# %%
""" Phase analysis (weird_nodes)"""
plt.close('all')

weird_nodes = (np.array(np.loadtxt('nodes_to_review_cons1.txt',delimiter=','))).astype(int)

t=np.linspace(0,x_hat_tikh0.shape[1]/fs,x_hat_tikh0.shape[1])
for i in range(0,len(weird_nodes)):
    signal_gt=x_ref_cons_full[weird_nodes[i],:]/np.max(x_ref_cons_full[weird_nodes[i],:])
    signal_tikh0=x_hat_tikh0[weird_nodes[i],:]/np.max(x_hat_tikh0[weird_nodes[i],:])
    signal_cons1=x_hat_cons1[weird_nodes[i],:]/np.max(x_hat_cons1[weird_nodes[i],:])

    instant_phase_gt = -np.arctan2(np.imag(sigproc.hilbert(signal_gt)),signal_gt)
    instant_phase_tikh0 = -np.arctan2(np.imag(sigproc.hilbert(signal_tikh0)),signal_tikh0)
    instant_phase_cons1 = -np.arctan2(np.imag(sigproc.hilbert(signal_cons1)),signal_cons1)

    pxx_gt_norm = pxx_ref[weird_nodes[i],:]/np.max(pxx_ref[weird_nodes[i],:])
    pxx_tikh0_norm = pxx_tikh0[weird_nodes[i],:]/np.max(pxx_tikh0[weird_nodes[i],:])
    pxx_cons1_norm = pxx_cons1[weird_nodes[i],:]/np.max(pxx_cons1[weird_nodes[i],:])

    plt.figure(),
    plt.subplot(221),plt.title('Signals. Node: %d.' %(weird_nodes[i]))
    plt.plot(t,signal_gt,'b'),
    plt.plot(t,signal_tikh0,'r'),plt.grid('on'),plt.xlim((1,7))
    plt.legend(('GT','Tikh0'))
    plt.subplot(222),    
    plt.plot(f,pxx_gt_norm,'b'),
    plt.plot(f_tikh0,pxx_tikh0_norm,'r'),plt.xlim((0,30)),plt.grid('on'),
    plt.title('DF_GT= %.3f Hz. DF_Tikh0=%.3f Hz.' %(df_ref[weird_nodes[i]],df_tikh0[weird_nodes[i]]))
    plt.legend(('GT','Tikh0'))
    
    plt.subplot(223),plt.title('Signals. Node: %d.' %(weird_nodes[i]))
    plt.plot(t,signal_gt,'b'),
    plt.plot(t,signal_cons1,'r'),plt.grid('on'),plt.xlim((1,7))
    plt.legend(('GT','Cons1'))
    plt.subplot(224),    
    plt.plot(f,pxx_gt_norm,'b'),
    plt.plot(f_cons1,pxx_cons1_norm,'r'),plt.xlim((0,30)),plt.grid('on'),
    plt.title('DF_GT= %.3f Hz. DF_Cons1=%.3f Hz.' %(df_ref[weird_nodes[i]],df_cons1[weird_nodes[i]]))
    plt.legend(('GT','Cons1'))
    
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()