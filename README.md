# PointNet++  reg #
 
## train ##
```
python train_reg.py --model pointnet2_reg  --log_dir pointnet2_reg
```

* 注意默认参数是1024个点的采样

## test ##

```
python test_reg.py   --log_dir pointnet2_reg
```

## infer ##

```
python infer_reg.py   --log_dir pointnet2_reg
```