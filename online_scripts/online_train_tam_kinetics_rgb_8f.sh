# 8 video clips per GPU
python -u online_main.py kinetics RGB --arch resnet50 \
--num_segments 8 --gd 20 --lr 0.01 --lr_steps 50 75 90 --epochs 100 --batch-size 8 \
-j 8 --dropout 0.5 --consensus_type=avg --root_log ./checkpoint/this_ckpt \
--root_model ./checkpoints/this_ckpt --eval-freq=1 --npb \
--tam  --dense_sample --wd 0.0001
