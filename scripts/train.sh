CUDA_VISIBLE_DEVICES=0
!python3 main.py  --epochs 15\
                --batch_size 1\
                --num_workers 2\
                --dataset scared\
                --dataset_directory ./sample_scared/
