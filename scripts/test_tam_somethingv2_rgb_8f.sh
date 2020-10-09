# test TAM on Something-Something V2
python -u test_models.py somethingv2 \
--weights=./checkpoints/somethingv2_RGB_resnet50_tam_avg_segment8_e50/ckpt.best.pth.tar \
--test_segments=8 --test_crops=3 \
--full_res --sample uniform-2 --batch_size 32
