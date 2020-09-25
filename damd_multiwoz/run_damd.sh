# CUDA_VISIBLE_DEVICES=5 python model.py -mode train -cfg seed=444 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False enable_aspn=True
# CUDA_VISIBLE_DEVICES=5 python model.py -mode train -cfg seed=443 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False enable_aspn=True
# CUDA_VISIBLE_DEVICES=3 python model.py -mode train -cfg seed=442 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False enable_aspn=True
# CUDA_VISIBLE_DEVICES=3 python model.py -mode train -cfg seed=556 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False
# CUDA_VISIBLE_DEVICES=3 python model.py -mode train -cfg seed=555 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False
# CUDA_VISIBLE_DEVICES=3 python model.py -mode train -cfg seed=557 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False
# CUDA_VISIBLE_DEVICES=3 python model.py -mode train -cfg seed=775 cuda_device=0 exp_no=aug_sample3 batch_size=60 multi_acts_training=True multi_act_sampling_num=3 enable_aspn=True

CUDA_VISIBLE_DEVICES=1 python model.py -mode train -cfg seed=557 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False
CUDA_VISIBLE_DEVICES=1 python model.py -mode train -cfg seed=557 cuda_device=0 exp_no=no_aug batch_size=128 multi_acts_training=False enable_aspn=True
CUDA_VISIBLE_DEVICES=1 python model.py -mode train -cfg seed=557 cuda_device=0 exp_no=aug_sample3 batch_size=60 multi_acts_training=True multi_act_sampling_num=3 enable_aspn=True
#CUDA_VISIBLE_DEVICES=2 python model.py -mode train -cfg seed=774 cuda_device=0 exp_no=aug_sample3 batch_size=60 multi_acts_training=True multi_act_sampling_num=3 enable_aspn=True
# CUDA_VISIBLE_DEVICES=0 python model.py -mode test -cfg cuda_device=0 eval_load_path=experiments/all_no_aug_sd441_lr0.005_bs128_sp5_dc3 aspn_decode_mode=greedy
# CUDA_VISIBLE_DEVICES=0 python model.py -mode test -cfg cuda_device=0 eval_load_path=experiments/all_no_aug_sd443_lr0.005_bs128_sp5_dc3 aspn_decode_mode=greedy
# CUDA_VISIBLE_DEVICES=0 python model.py -mode test -cfg cuda_device=0 eval_load_path=experiments/all_no_aug_sd444_lr0.005_bs128_sp5_dc3 aspn_decode_mode=greedy
# CUDA_VISIBLE_DEVICES=0 python model.py -mode test -cfg cuda_device=0 eval_load_path=experiments/all_no_aug_sd556_lr0.005_bs128_sp5_dc3 aspn_decode_mode=greedy
# CUDA_VISIBLE_DEVICES=0 python model.py -mode test -cfg cuda_device=0 eval_load_path=experiments/all_no_aug_sd557_lr0.005_bs128_sp5_dc3 aspn_decode_mode=greedy
# CUDA_VISIBLE_DEVICES=0 python model.py -mode test -cfg cuda_device=0 eval_load_path=experiments/all_no_aug_sd558_lr0.005_bs128_sp5_dc3 aspn_decode_mode=greedy
# CUDA_VISIBLE_DEVICES=0 python model.py -mode test -cfg cuda_device=0 eval_load_path=experiments/all_aug_sample3_sd774_lr0.005_bs60_sp5_dc3 aspn_decode_mode=greedy
# CUDA_VISIBLE_DEVICES=0 python model.py -mode test -cfg cuda_device=0 eval_load_path=experiments/all_aug_sample3_sd775_lr0.005_bs60_sp5_dc3 aspn_decode_mode=greedy
# CUDA_VISIBLE_DEVICES=0 python model.py -mode test -cfg cuda_device=0 eval_load_path=experiments/all_aug_sample3_sd776_lr0.005_bs60_sp5_dc3 aspn_decode_mode=greedy

