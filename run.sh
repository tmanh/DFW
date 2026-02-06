python water_level.py --cfg=config/idw.yaml
python water_level.py --cfg=config/mlpw_48.yaml --training
python water_level.py --cfg=config/ok_disp.yaml

python ked.py

python rgwater.py
python nngp.py --training

python rgwater.py --ckpt=model_rgnn_t20.pth --top=0.2
# python rgwater.py --ckpt=model_rgnn_t40.pth --top=0.4
# python rgwater.py --ckpt=model_rgnn_t60.pth --top=0.6
# python rgwater.py --ckpt=model_rgnn_t80.pth --top=0.8
# python rgwater.py --ckpt=model_rgnn_t100.pth --top=1.0

# python rgwater.py --ckpt=model_rgnn_tidal.pth --top=0.2
# python rgwater.py --ckpt=model_rgnn_d.pth --top=0.2
# python rgwater.py --ckpt=model_rgnn_ns.pth --top=0.2
# python rgwater.py --ckpt=model_rgnn_ne.pth --top=0.2
# python rgwater.py --ckpt=model_rgnn_repeat.pth --top=0.2