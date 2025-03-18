#Diffusion Policy Mujoco
train:
python train.py --config-dir=. --config-name=train_diffusion_unet_real_image_workspace_Mujoco.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

