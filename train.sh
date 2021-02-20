CUDA_VISIBLE_DEVICES=0,1,2,3 /data4/jiangjianping/anaconda3/envs/nerf/bin/python run_nerf.py --config configs/fern.txt --ft_path=./logs/fern_test/050000.tar
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_nerf.py --config configs/fern_test_origin.txt --expname fern_exp1 --dataset_type llff
CUDA_VISIBLE_DEVICES=4,5,6,7 python run_nerf.py --config configs/fern_test_origin.txt --expname fern_exp2 --dataset_type llff --netdepth 6 --netdepth_fine 6 --skip 3
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_nerf.py --config configs/fern_test_origin.txt --expname fern_exp3 --dataset_type llff --netwidth 128 --netwidth_fine 128
CUDA_VISIBLE_DEVICES=4,5,6,7 python run_nerf.py --config configs/fern_test_origin.txt --expname fern_exp4 --dataset_type llff --netdepth 6 --netdepth_fine 6 --skip 3 --netwidth 128 --netwidth_fine 128
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_nerf.py --config configs/fern_test_origin.txt --expname fern_exp5 --dataset_type llff_gray --is_grayrgb
CUDA_VISIBLE_DEVICES=4,5,6,7 python run_nerf.py --config configs/fern_test_origin.txt --expname fern_exp6 --dataset_type llff_gray --is_grayrgb --netdepth 6 --netdepth_fine 6 --skip 3
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_nerf.py --config configs/fern_test_origin.txt --expname fern_exp7 --dataset_type llff_gray --is_grayrgb --netwidth 128 --netwidth_fine 128
CUDA_VISIBLE_DEVICES=4,5,6,7 python run_nerf.py --config configs/fern_test_origin.txt --expname fern_exp8 --dataset_type llff_gray --is_grayrgb --netdepth 6 --netdepth_fine 6 --skip 3 --netwidth 128 --netwidth_fine 128

CUDA_VISIBLE_DEVICES=4,5,6,7 python run_nerf.py --config configs/fern_test_origin.txt --expname fern_exp14 --dataset_type llff_gray --is_grayrgb --netdepth 12 --netdepth_fine 12 --skip 6
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_nerf.py --config configs/fern_test_origin.txt --expname fern_exp15 --dataset_type llff_gray --is_grayrgb --netwidth 512 --netwidth_fine 512

CUDA_VISIBLE_DEVICES=4,5,6,7 python run_nerf.py --config configs/fern_test_origin.txt --expname fern_exp16 --N_rand 2048 --dataset_type llff_gray --is_grayrgb --netdepth 12 --netdepth_fine 12 --skip 6 --netwidth 512 --netwidth_fine 512
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_nerf.py --config configs/fern_test_origin.txt --expname fern_exp12 --N_rand 4096 --dataset_type llff_gray --is_grayrgb --multires 20 --multires_views 8
CUDA_VISIBLE_DEVICES=4,5,6,7 python run_nerf.py --config configs/fern_test_origin.txt --expname fern_exp18 --N_rand 4096 --dataset_type llff_gray --is_grayrgb --multires 30 --multires_views 10

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_nerf.py --config configs/fern_test_origin.txt --expname fern_exp19 --N_rand 2048 --dataset_type llff_gray --is_grayrgb --multires 20 --multires_views 8 --netdepth 12 --netdepth_fine 12 --skip 6 --netwidth 512 --netwidth_fine 512