import os

config = {
    'model_name': 'vaeaqp',
    'input_file': 'orders.csv',
    'output_dir': 'outputs/{}/'.format('vaeaqp'),
    'data_output_dir': 'outputs/{}/'.format('vae'),
    'batch_size': 512,
    'latent_dim': 64,
    'neuron_list': 256,
    'epochs': 100,
    'rejection':1,
    'seed':42,
    'gpus':0
}
cmd = "python train.py "
for key in config:
    cmd += "--{} {} ".format(key,config[key])
print(cmd)
os.system(cmd)
