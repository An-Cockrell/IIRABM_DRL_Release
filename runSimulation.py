import ctypes
import numpy as np
import wrapper_setup  # Ensure this correctly exposes setUpWrapper
import sys

MAX_OXYDEF = 8160 #Oxydef level where subject is considered dead
HEAL_OXYDEF = 50 #Oxydef level where subject is considered totally healed
MAX_STEPS = 20000
MIN_STEPS = 100


def createIIRABM(OH, IS, NRI, NIR, injNum, seed, numCytokines=9, abxMult=-1.0, SIM=None, ruleMatrix='baseParameterization2.npy'):
    if SIM is None:
        SIM = wrapper_setup.setUpWrapper()

    oxyHeal = ctypes.c_float(OH)
    IS = ctypes.c_int(IS)
    NRI = ctypes.c_int(NRI)
    NIR = ctypes.c_int(NIR)
    injNum = ctypes.c_int(injNum)
    seed = ctypes.c_int(seed)
    numCytokines = ctypes.c_int(numCytokines)

    internalParam = np.load(ruleMatrix)
    rank = ctypes.c_int(0)
    numMatrixElements = 432
    array_type = ctypes.c_float * numMatrixElements
    antibioticMultiplier = ctypes.c_float(abxMult)

    instance = SIM.CreateInstance(oxyHeal, IS, NRI, NIR, injNum, seed, numMatrixElements, array_type(*internalParam),
                                  rank, antibioticMultiplier)
    return instance

def calculate_done(oxydef):
    DONE = False
    if oxydef < HEAL_OXYDEF:
        DONE = True
    if oxydef> MAX_OXYDEF:
        DONE = True
    return DONE

def run_simulation_and_get_signals(OH, IS, NRI, NIR, injNum, seed, numCytokines=9, abxMult=-1.0, ruleMatrix='baseParameterization2.npy'):
    SIM = wrapper_setup.setUpWrapper()
    simulation_instance = createIIRABM(OH, IS, NRI, NIR, injNum, seed, numCytokines, abxMult, SIM, ruleMatrix)

    i = 0
    done = False
    OD=np.zeros(MAX_STEPS)
    while (i <= MAX_STEPS) and (done is False):

        SIM.singleStep(simulation_instance)
        oxydef = SIM.getAllSignalsReturn(simulation_instance)[0, i]

        if i > MIN_STEPS:
            done = calculate_done(oxydef)
#        print(i, oxydef)
        OD[i]=oxydef

        i += 1

    # Retrieve all signals //NOT WORKING
#    all_signals = SIM.getAllSignalsReturn(simulation_instance)  # Or call a method if needed

#    return all_signals
#    print("OD=",OD)
    return OD

# if __name__ == '__main__':
#     _OH = 0.08
#     _IS = 2
#     _NRI = 0
#     _NIR = 2
#     _injNum = 20
#     seed = 1234
#     all_signals = run_simulation_and_get_signals(_OH, _IS, _NRI, _NIR, _injNum, seed)


# numStochasticReplicates=50;
# params=[]
# q=0
# for i_oh in range(8):
#     for i_is in range(4):
#         for i_nri in range(6):
#             for i_nir in range (4):
#                 for i_inj in range(25):
#                     oh=0.06+i_oh*0.02
#                     ins=1+i_is
#                     nri=i_nri
#                     nir=i_nir+1
#                     inj=i_inj+1
#                     params.append([oh,ins,nri,nir,inj])
#
# params=np.asarray(params)
#
# np.save('Params.npy',params)

AllResults=[]
p=[]
params=np.load('Params.npy')

numStochasticReplicates=50;

# print ('argument list', sys.argv)

bottom=int(sys.argv[1])
top=int(sys.argv[2])
rfile=str(sys.argv[3])
pfile=str(sys.argv[4])

# print(rfile)
# print(pfile)


for j in range(bottom,top):
    print(j)
    oxyHeal=params[j,0]
    infspr=int(params[j,1])
    numRecurInj=int(params[j,2])
    nir=int(params[j,3])
    inj=int(params[j,4])
    seed=0
    for i in range(numStochasticReplicates):
#        print(i)
        result=run_simulation_and_get_signals(oxyHeal,infspr,numRecurInj,nir,inj,seed+i)
#    tempResults[i*20:(i+1)*20,:]=result
#    AllResults=np.vstack((AllResults,tempResults))
        AllResults.append(result)
        p.append([oxyHeal,infspr,numRecurInj,nir,inj,seed+i])
        np.save(rfile,np.asarray(AllResults))
        np.save(pfile,np.asarray(p))

print(AllResults)
