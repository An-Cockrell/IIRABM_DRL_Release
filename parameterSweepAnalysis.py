import ctypes
import numpy as np
import wrapper_setup  # Ensure this correctly exposes setUpWrapper
import sys


sweep_params1=np.load('Params1.npy')
sweep_results1=np.load('Result1.npy')

for i in range(2,10):
    rfile=str('Result')+str(i)+str('.npy')
    pfile=str('Params')+str(i)+str('.npy')
    print(rfile,pfile)
    rtemp=np.load(rfile)
    ptemp=np.load(pfile)
    sweep_params1=np.vstack((sweep_params1,ptemp))
    sweep_results1=np.vstack((sweep_results1,rtemp))

np.save('sweep_results1.npy',sweep_results1)
np.save('sweep_params1.npy',sweep_params1)

sweep_params2=np.load('Params10.npy')
sweep_results2=np.load('Result10.npy')

for i in range(11,20):
    rfile=str('Result')+str(i)+str('.npy')
    pfile=str('Params')+str(i)+str('.npy')
    print(rfile,pfile)
    rtemp=np.load(rfile)
    ptemp=np.load(pfile)
    sweep_params2=np.vstack((sweep_params2,ptemp))
    sweep_results2=np.vstack((sweep_results2,rtemp))

np.save('sweep_results2.npy',sweep_results2)
np.save('sweep_params2.npy',sweep_params2)


sweep_params3=np.load('Params20.npy')
sweep_results3=np.load('Result20.npy')

for i in range(21,40):
    rfile=str('Result')+str(i)+str('.npy')
    pfile=str('Params')+str(i)+str('.npy')
    print(rfile,pfile)
    rtemp=np.load(rfile)
    ptemp=np.load(pfile)
    sweep_params3=np.vstack((sweep_params3,ptemp))
    sweep_results3=np.vstack((sweep_results3,rtemp))

np.save('sweep_results3.npy',sweep_results3)
np.save('sweep_params3.npy',sweep_params3)
