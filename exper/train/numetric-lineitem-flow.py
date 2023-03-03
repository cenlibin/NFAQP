import sys, os
sys.path.append('/home/clb/AQP')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import shutil
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tensorboardX import SummaryWriter
from utils  import DataPrefetcher, discretize_dataset, DataWrapper
from utils import *
from datasets import get_dataset_from_name

# runtime config
SEED = 1638128

# training hyperparameters
dataset_name = 'lineitem-numetric'
model_tag = 'flow'

learning_rate = 0.005
save_interval = -1 # -1 mean not interval save
eval_interval = 2500
grad_norm_clip_value = 5.
train_batch_size = 512
val_batch_size = train_batch_size * 4
num_training_steps = 400000

device = torch.device("cuda")

model_config = {
    'num_features': None,
    'num_bins': 8,
    'num_hidden_features': 56,
    'num_transform_blocks': 2,
    'num_flow_steps': 6,
    'dropout_probability': 0.0,
    'tail_bound': 3,
    'use_batch_norm': False,
    'base_transform_type': 'rq_coupling',
    'linear_transform_type': 'lu'
}

save_dir = f'{dataset_name}-{model_tag}'
seed_everything(SEED)

def train():
    # init model and result saving path
    save_path = os.path.join(MODELS_DIR, save_dir)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    print(f"saving to {save_path}")
    summer_writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    # Load Data
    T = TimeTracker()
    train_set, val_set = get_dataset_from_name(dataset_name)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator(device=device)
        # pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        generator=torch.Generator(device=device)
        # pin_memory=True
    )
    T.reportIntervalTime('Load data loader')
    epochs = num_training_steps // len(train_loader) + 1
    print(f'total training steps:{num_training_steps} total epochs:{epochs}')

    model_config['num_features'] = train_set[0].shape[0]
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
    epoch = 0
    model.train()
    while global_step < num_training_steps:
        pbar = tqdm(total=len(train_loader))
        train_prefetcher = DataPrefetcher(train_loader)
        train_batch = train_prefetcher.next()
        epoch += 1
        while train_batch is not None:
            optimizer.zero_grad()
            log_density = model.log_prob(train_batch)
            loss = -torch.mean(log_density)  # maximize negative log likelyhood
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_norm_clip_value)
            optimizer.step()
            summer_writer.add_scalar('loss', loss.detach(), global_step)
            pbar.set_description_str(f'epoch:{epoch}/{epochs} step:{global_step}/{num_training_steps}({100.0 * global_step / num_training_steps:.3f}%) loss:{loss.item():.3f}')
            pbar.update()
            global_step += 1
            train_batch = train_prefetcher.next()
            # ====================================interval save============================================
            if save_interval != -1 and (global_step + 1) % save_interval == 0:
                model.eval()
                tqdm.write(f'saving for step {global_step}, best val score is {best_val_score}')
                torch.save(model, os.path.join(f'step-{global_step}.pt'))
                model.train()

            # ====================================valid===========================================
            if eval_interval != -1 and (global_step + 1) % eval_interval == 1:
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
                    print(f'epoch:{epoch} val log density is {val_logp:.4f}')
                    summer_writer.add_scalar('evaluate mean log probability density', val_logp, epoch)

                # ====================================save============================================
                if val_logp > best_val_score:
                    tqdm.write(f'New best score:{val_logp:.3f} last best score:{best_val_score:.3f} !')
                    best_val_score = val_logp
                    torch.save(model, os.path.join(save_path, 'best.pt'))
                
        # ================================lr schedule==========================================
        model.train()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print("schedule lr to {}".format(lr))
        summer_writer.add_scalar("lr", lr, epoch)



if __name__ == '__main__':
    train()
