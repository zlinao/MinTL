from joblib import Parallel, delayed
import queue
import os
import time

# Define number of GPUs available
GPU_available = [5]
N_GPU = len(GPU_available)

experiments = [  
    #"python model.py -mode train -cfg seed=557 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False",
#                 "python model.py -mode train -cfg seed=559 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False",
              "python model.py -mode train -cfg seed=557 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False enable_aspn=True",
            #     "python model.py -mode train -cfg seed=559 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False enable_aspn=True",
            #  "python model.py -mode train -cfg seed=560 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False enable_aspn=True",
            "python model.py -mode train -cfg seed=557 cuda_device=0 exp_no=aug_sample3 batch_size=40 multi_acts_training=True multi_act_sampling_num=3 enable_aspn=True",
            #"python model.py -mode train -cfg seed=558 cuda_device=0 exp_no=aug_sample3 batch_size=40 multi_acts_training=True multi_act_sampling_num=3 enable_aspn=True",
                ]

q = queue.Queue(maxsize=N_GPU)
mapper = {}
invert_mapper = {}
for i in range(N_GPU):
    mapper[i] = GPU_available[i]
    invert_mapper[GPU_available[i]] = i
    q.put(i)

def runner(cmd):
    gpu = mapper[q.get()]
    os.system("CUDA_VISIBLE_DEVICES=%d %s" % (gpu, cmd))
    q.put(invert_mapper[gpu])

# Change loop
Parallel(n_jobs=N_GPU, backend="threading")( delayed(runner)(e) for e in experiments)