# test TAM on Something-Something V1
python test_models.py something \
--weights=./checkpoints/something_RGB_resnet50_tam_avg_segment8_e50/ckpt.best.pth.tar \
--test_segments=8 --test_crops=1 \
--sample uniform-1 --batch_size 64
