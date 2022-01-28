#------------------------------ 
python3 main.py --gpus 0,1,2,3 --train t --model vgg19 --total_epochs 10000 --exp_name check
nohup python3 -u main.py --gpus 0,1,2,3 --train t --model vgg19 --total_epochs 2000 --port 1234 --exp_name esc50 > log/esc50.log &
nohup python3 -u main.py --gpus 0,1,2,3 --train t --masked_train f --model vgg19 --total_epochs 2000 --port 2345 --exp_name esc50 > log/esc50.log &
nohup python3 -u main.py --gpus 0,1,2,3 --train t --resume t --load_epoch 2000 --model vgg19  --total_epochs 10000 --exp_name check > log/check.log &
#------------------------------ 
python3 main.py --gpus 12 --test t --model vgg19 --load_epoch 1000  --exp_name esc50 --power 0.4 --gain 50 --port 4321
python3 single_test.py --gpu 4 --model vgg19 --load_epoch 3000 --exp_name LibDem-vgg19-vad --power 0.4 --gain 50
#------------------------------ 
