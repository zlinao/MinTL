

"""
T5:
end2end: "python train.py --mode train --context_window 2 --pretrained_checkpoint t5-small --cfg seed=557 batch_size=32",
         "python train.py --mode train --context_window 2 --gradient_accumulation_steps 8 --pretrained_checkpoint t5-base --cfg seed=557 batch_size=8",
         
DST:  "python DST.py --mode train --context_window 3 --cfg seed=557 batch_size=32",
        "python DST.py --mode train --context_window 3 --gradient_accumulation_steps 5 --pretrained_checkpoint t5-base --cfg seed=557 batch_size=12",
        "python DST.py --mode train --context_window 5 --version 2.1 --cfg seed=557 batch_size=32",
        "python DST.py --mode train --context_window 5 --version 2.1 --gradient_accumulation_steps 5 --pretrained_checkpoint t5-base --cfg seed=557 batch_size=12",
Lexicalize:
python train.py --mode relex --context_window 2 --pretrained_checkpoint t5-small --cfg seed=557 batch_size=32 --model_path experiments/all_sd557_lr0.0006_bs32_sp5_dc0.8_cw2_model_t5-small_noupdateFalse_1.0 --device cpu
python train.py --mode relex --context_window 2 --gradient_accumulation_steps 8 --pretrained_checkpoint t5-base --cfg seed=557 batch_size=8 --model_path experiments/all_sd557_lr0.0006_bs8_sp5_dc0.8_cw2_model_t5-base_noupdateFalse_1.0 --device cpu



BART:
end2end: "python train.py --mode train --context_window 2 --pretrained_checkpoint bart-large-cnn --gradient_accumulation_steps 8 --lr 3e-5 --back_bone bart --cfg seed=557 batch_size=8",
DST: "python DST.py --mode train --context_window 3 --gradient_accumulation_steps 10 --pretrained_checkpoint bart-large-cnn --back_bone bart --lr 1e-5 --cfg seed=557 batch_size=4",
 "python DST.py --mode train --context_window 5 --version 2.1 --gradient_accumulation_steps 10 --pretrained_checkpoint bart-large-cnn --back_bone bart --lr 1e-5 --cfg seed=557 batch_size=4",
Lexicalize:
python train.py --mode relex --context_window 2 --pretrained_checkpoint bart-large-cnn --gradient_accumulation_steps 8 --lr 2e-5 --back_bone bart --cfg seed=557 batch_size=8 --model_path experiments/all_sd557_lr3e-05_bs8_sp5_dc0.8_cw2_model_bart-large-cnn_noupdateFalse_1.0 --device cpu
"""