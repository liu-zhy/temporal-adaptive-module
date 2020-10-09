# 16 video clips per GPU
python main.py something RGB --arch resnet50 \
--num_segments 8 --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 --batch-size 64 \
-j 2 --dropout 0.6 --consensus_type=avg --root_log ./checkpoints/this_ckpt \
--root_model ./checkpoints/this_ckpt --eval-freq=1 --npb \
--tam --wd 0.001
