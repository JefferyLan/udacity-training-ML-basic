
#session 1:
#  python train.py data_directory
#  在训练网络时，输出训练损失、验证损失和验证准确率
#session 2:
# 设置保存检查点的目录：python train.py data_dir --save_dir save_directory
# 选择架构：python train.py data_dir --arch "vgg13"
# 设置超参数：python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# 使用 GPU 进行训练：python train.py data_dir --gpu

# session 3:
# python predict.py input checkpoint
# 选项：
# 返回前 KK 个类别：python predict.py input checkpoint --top_k 3
# 使用类别到真实名称的映射：python predict.py input checkpoint --category_names cat_to_name.json
# 使用 GPU 进行训练：python predict.py input checkpoint --gpu
# 结束


python train.py ./flowers/train --save_dir ./models/vgg16-checkpoint.tar --valid_dir ./flowers/valid --arch vgg16 --learning_rate 0.01 --hidden_units 2048 --epochs 1 --gpu 0

python predict.py ./flowers/test/12/image_03994.jpg  ./models/vgg16-checkpoint.tar --gpu 0 --category_names cat_to_name.json --top_k 10
