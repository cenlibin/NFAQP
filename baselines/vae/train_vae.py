from argparse import ArgumentParser
import time
from utils import load_table
from VAE import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
sys.path.append('/home/clb/AQP')
from utils import get_model_size_mb
DATASET_NAME = 'pm25'
ROWS_THRESHOLD = 500000

parser = ArgumentParser(description='VAE')
parser.add_argument('--model_name', type=str, action='store', default='VAE')
parser.add_argument('--input_file', type=str, action='store', default=f'/home/clb/AQP/data/{DATASET_NAME}.csv')
parser.add_argument('--output_dir', type=str, action='store', default='outputs/{}/'.format('vae-orders'))
parser.add_argument('--data_output_dir', type=str, action='store', default='outputs/{}/'.format('vae-orders'))
parser.add_argument('--batch_size', type=int, action='store', default=256)
parser.add_argument('--latent_dim', type=int, action='store', default=256)
parser.add_argument('--neuron_list', type=int, action='store', default=200, help='Latent Dimension size Default: 200.')
parser.add_argument('--epochs', type=int, action='store', default=300)
parser.add_argument('--log_interval', type=int, action='store', default=25)
parser.add_argument('--rejection', type=int, action='store', default=1)
parser.add_argument('--num_samples', type=int, action='store', default=10000)
parser.add_argument('--seed', type=int, action='store', default=42)
parser.add_argument('--gpus', type=str, action='store', default='0')
args = parser.parse_args()
OUT_ROOT = f"/home/clb/AQP/output/VAE-{DATASET_NAME}/"
args.output_dir = OUT_ROOT
args.data_output_dir = OUT_ROOT
args.model_name = 'VAE'

# Parsing and saving the input arguments

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

if os.path.exists(OUT_ROOT):
    import shutil
    shutil.rmtree(OUT_ROOT)
# Create output dir
os.makedirs(OUT_ROOT, exist_ok=True)

# Set the seeds. Default: 42
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Specify model parameters
model_name = args.model_name
file_name = f'/home/clb/AQP/data/{DATASET_NAME}.csv'
args.input_file = file_name
batch_size = args.batch_size
epochs = args.epochs
log_interval = args.log_interval
latent_dim = args.latent_dim
neuron_list = args.neuron_list

# Specify the size of synthetic dataset size
num_instance = args.num_samples

# Specify which are categorical and which are numeric
# We don't care about header, so delete the header from input if it has.
# After reading the input, this cell converts the input to encoding.
print("Reading INPUT File")
orig_df = load_table(DATASET_NAME)
orig_df = orig_df.dropna().reset_index(drop=True)
if orig_df.shape[0] >= ROWS_THRESHOLD:
    df = orig_df.sample(ROWS_THRESHOLD).reset_index(drop=True)
else:
    df = orig_df
print("Original", orig_df.shape)
print("Sampled", df.shape)
cols = df.columns
ctype = df.dtypes
cat_cols = list(filter(lambda x: ctype[x] == 'object', cols))
num_cols = list(filter(lambda x: ctype[x] != 'object', cols))
org_input_dim = len(cat_cols) + len(num_cols)

# hot_hot, num_num and hot_num three supported encoding type for categorical_numerical data
encoding_type = "hot_num"  # categorical hot and numeric column in numeric form

print("Transforming Train/Test")
if os.path.exists(os.path.join(OUT_ROOT, 'data.pkl')):
# if os.path.exists(args.data_output_dir+'data.pkl'):
    x_train, x_test = pickle.load(open(os.path.join(OUT_ROOT, 'data.pkl'), 'rb'))
else:
    x_train, x_test = transform_forward(args, df, num_cols, cat_cols, encoding_type)
    pickle.dump((x_train, x_test), open(os.path.join(OUT_ROOT, 'data.pkl'), 'wb'), protocol=4)
print("Train Shape: ", x_train.shape, "Test Shape", x_test.shape)

use_cuda = torch.cuda.is_available()
if use_cuda:
    dev_gpu = "cuda"  # Yes, it uses a GPU.
    device_gpu = torch.device(dev_gpu)

start = time.time()

model_name = args.model_name
model = VAE(x_train.shape[1], latent_dim, neuron_list)
print(f'model size for VAE with dataset {DATASET_NAME} is {get_model_size_mb(model)} mb!')
if use_cuda:
    model.to(device_gpu)
    x_train = torch.from_numpy(x_train).to(device_gpu)
    x_test = torch.from_numpy(x_test).to(device_gpu)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss = []
bestLoss = -1
start_epoch = 0
epochs = args.epochs
if os.path.exists(args.output_dir + 'model_state'):
    checkpoint = torch.load(args.output_dir+'model_state')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    bestLoss = checkpoint['bestLoss']
    loss = pickle.load(open(args.output_dir+'loss.pkl', 'rb'))

for epoch in tqdm(range(start_epoch + 1, epochs + 1)):
    currentLoss = train(model, optimizer, epoch, x_train,
                        log_interval, batch_size, org_input_dim, args.rejection)
    loss.append(currentLoss)
    if bestLoss < 0 or (currentLoss < bestLoss):
        torch.save(model, args.output_dir+'model.pt')
        bestLoss = currentLoss

    if epoch % 10 == 0:
        # The value of T computed here is used during sample generation.
        t_val = calculate_t(model, x_train, batch_size)
        time_taken = time.time()-start

        pickle.dump(t_val, open(args.output_dir+'t-val.pkl', 'wb'))
        pickle.dump(loss, open(args.output_dir+'loss.pkl', 'wb'))
        pickle.dump(time_taken, open(args.output_dir+'time_taken.pkl', 'wb'))
        plt.plot(loss)
        plt.savefig(args.output_dir+'loss.png')

        state = {'epoch': epoch, 'state_dict': model.state_dict(
        ), 'optimizer': optimizer.state_dict(), 'bestLoss': bestLoss}
        torch.save(state, args.output_dir+'model_state')

# The value of T computed here is used during sample generation.
t_val = calculate_t(model, x_train, batch_size)
print("90th Percentile value of T", t_val)
print("Training Ends")
time_taken = time.time()-start

pickle.dump(t_val, open(args.output_dir+'t-val.pkl', 'wb'))
pickle.dump(loss, open(args.output_dir+'loss.pkl', 'wb'))
pickle.dump(time_taken, open(args.output_dir+'time_taken.pkl', 'wb'))
plt.plot(loss)
plt.savefig(args.output_dir+'loss.png')

state = {'epoch': epochs, 'state_dict': model.state_dict(
), 'optimizer': optimizer.state_dict(), 'bestLoss': bestLoss}
torch.save(state, args.output_dir+'model_state')
