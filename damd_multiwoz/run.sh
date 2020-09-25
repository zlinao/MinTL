# CUDA_VISIBLE_DEVICES=2 python model.py -mode train -cfg seed=444 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False enable_aspn=True
# CUDA_VISIBLE_DEVICES=7 python model.py -mode train -cfg seed=559 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False 
# CUDA_VISIBLE_DEVICES=5 python model.py -mode train -cfg seed=776 cuda_device=0 exp_no=aug_sample3 batch_size=60 multi_acts_training=True multi_act_sampling_num=3 enable_aspn=True
# CUDA_VISIBLE_DEVICES=5 python model.py -mode train -cfg seed=775 cuda_device=0 exp_no=aug_sample3 batch_size=60 multi_acts_training=True multi_act_sampling_num=3 enable_aspn=True
# CUDA_VISIBLE_DEVICES=6 python model.py -mode train -cfg seed=558 cuda_device=0 exp_no=no_aug batch_size=1 multi_acts_training=False
# CUDA_VISIBLE_DEVICES=5 python model.py -mode test -cfg cuda_device=0 eval_load_path=experiments/all_aug_sample3_sd776_lr0.005_bs60_sp5_dc3 aspn_decode_mode=greedy batch_size=1
# python model.py -mode test -cfg cuda_device=0 eval_load_path=$path aspn_decode_mode=greedy
# #transferlm
# export PYTHONPATH=/home/lin/transferlm/damd_multiwoz:/home/lin/transferlm/damd_multiwoz/data:/home/lin/transferlm/damd_multiwoz/db:$PYTHONPATH
# CUDA_VISIBLE_DEVICES=1 python train.py --mode train --cfg seed=557 batch_size=16 multi_acts_training=False
# CUDA_VISIBLE_DEVICES=1 python train.py --mode test --cfg seed=557 batch_size=2 --model_path experiments/all_sd557_lr6.25e-05_bs16_sp5_dc3
# CUDA_VISIBLE_DEVICES=3 python train.py --mode test --context_window 3 --cfg seed=557 batch_size=16 --model_path experiments/all_sd557_lr0.001_bs16_sp5_dc3_cw3_modelt5-small

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.0005 --mode train --cfg seed=557 batch_size=16 multi_acts_training=False
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.005 --mode train --cfg seed=557 batch_size=16 multi_acts_training=False
# export PYTHONPATH=$PYTHONPATH:/home/ershisui
CUDA_VISIBLE_DEVICES=7 python model.py -mode train -cfg seed=558 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False
CUDA_VISIBLE_DEVICES=7 python model.py -mode train -cfg seed=559 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False
CUDA_VISIBLE_DEVICES=7 python model.py -mode train -cfg seed=557 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False enable_aspn=True