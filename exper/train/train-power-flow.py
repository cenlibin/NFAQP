import datetime
import shutil
from nflows import utils
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys, os
sys.path.append('/home/clb/AQP')
from tensorboardX import SummaryWriter
from utils  import DataPrefetcher, discretize_dataset, DataWrapper
from utils import *

from datasets import get_dataset_from_name

# runtime config
SEED = 10086
GPU_ID = 1

# training hyperparameters
dataset_name = 'power'
model_tag = 'flow'

learning_rate = 0.001
monitor_interval = 5000
save_interval = -1 # -1 mean not interval save
grad_norm_clip_value = 5.
train_batch_size = 512
val_batch_size = train_batch_size * 4
epochs = 50
device = torch.device("cuda")

model_config = {
    'num_features': None,
    'num_bins': 8,
    'num_hidden_features': 64,
    'num_transform_blocks': 2,
    'num_flow_steps': 3,
    'dropout_probability': 0.0,
    'tail_bound': 3,
    'use_batch_norm': True,
    'base_transform_type': 'rq_coupling',
    'linear_transform_type': 'lu'
}

seed_everything(SEED)
set_gpu(GPU_ID)
save_dir = f'{dataset_name}-{model_tag}'


def train():
    # init model and result saving path
    save_path = os.path.join(OUTPUT_ROOT, save_dir)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    print(f"saving to {save_path}")
    summer_writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    # Load Data
    T = TimeTracker()
    dataset = get_dataset_from_name(dataset_name)
    train_loader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        # pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=dataset,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        # pin_memory=True
    )
    T.reportIntervalTime('Load data loader')
    total_training_steps = epochs * len(train_loader)
    print(f'total training steps:{total_training_steps}')

    model_config['num_features'] = dataset[0].shape[0]
    distribution = distributions.StandardNormal((model_config['num_features'],))
    transform = create_transform(model_config)
    model = flows.Flow(transform, distribution).to(device).train()
    print(f'num paramaters of {model_tag}:{get_model_size_mb(model):.5f} MB')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0)
    T.reportIntervalTime('create & setup model')

    # ====================================train===========================================
    best_val_score = -1e10
    global_step = 0
    model.train()
    for epoch in range(epochs):
        step = 0
        pbar = tqdm(total=len(train_loader))
        train_prefetcher = DataPrefetcher(train_loader)
        train_batch = train_prefetcher.next()
        while train_batch is not None:
            optimizer.zero_grad()
            log_density = model.log_prob(train_batch)
            loss = -torch.mean(log_density)  # maximize negative log likelyhood
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_norm_clip_value)
            optimizer.step()
            summer_writer.add_scalar('loss', loss.detach(), global_step)
            pbar.set_description_str(f'epochs:{epoch}/{epochs} step:{step} global step:{global_step} loss:{loss.item():.3f}')
            pbar.update()
            # ====================================interval save============================================
            if save_interval != -1 and (global_step + 1) % save_interval == 0:
                model.eval()
                tqdm.write(f'saving for step {global_step}, best val score is {best_val_score}')
                torch.save(model, os.path.join(f'step-{global_step}.pt'))
                model.train()

            global_step += 1
            step += 1
            train_batch = train_prefetcher.next()

        # ====================================valid===========================================
        model.eval()
        
        with torch.no_grad():
            val_logp, n_val_items = 0, 0
            val_bar = tqdm(total=len(val_loader), desc=f'evaluating for epoch {epoch}')
            val_prefetcher = DataPrefetcher(val_loader)
            val_batch = val_prefetcher.next()
            while val_batch is not None:
                n_val_items += val_batch.shape[0]
                val_logp += model.log_prob(val_batch).detach().sum()
                val_batch = val_prefetcher.next()
                val_bar.update()
            val_logp /= n_val_items
            tqdm.write(f'epoch:{epoch} val log density is {val_logp:.4f}')
            summer_writer.add_scalar('evaluate mean log probability density', val_logp, epoch)

        # ====================================save============================================
        if val_logp > best_val_score:
            tqdm.write(f'New best score:{val_logp:.3f} last best score:{best_val_score:.3f} !')
            best_val_score = val_logp
            torch.save(model, os.path.join(save_path, f'{dataset_name}-best.pt'))
            
        # ================================lr schedule==========================================
        model.train()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print("schedule lr to {}".format(lr))
        summer_writer.add_scalar("lr", lr, epoch)



if __name__ == '__main__':
    train()
