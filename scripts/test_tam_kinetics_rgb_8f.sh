# test TAM on Kinetics-400
python -u test_models.py kinetics \
--weights=./checkpoints/kinetics_RGB_resnet50_tam_avg_segment8_e100_dense/ckpt.best.pth.tar \
--test_segments=8 --test_crops=3 \
--full_res --sample dense-10 --batch_size 8
