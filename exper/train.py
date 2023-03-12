import sys
import os
sys.path.append('/home/clb/AQP')
from model_config import default_configs
from datasets import get_dataset_from_named
from utils import *
from utils  import DataPrefetcher, discretize_dataset
from table_wapper import TableWrapper
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch import optim
import shutil
import logging
import wandb
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="order")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model_size', type=str, default='small')
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--dequan', type=str, default="spline")


args = parser.parse_args()

# runtime config
SEED = 1638128
# training hyperparameters
DATASET_NAME = args.dataset
MODEL_TAG = f'flow-{args.model_size}'
DEQUAN_TYPE = args.dequan
GPU = args.gpu
LR = args.lr
MISSION_TAG = f'{MODEL_TAG}-{DATASET_NAME}-{DEQUAN_TYPE}'
SAVE_DIR = os.path.join(OUTPUT_ROOT, MISSION_TAG)
seed_everything(SEED)

save_interval = -1  # -1 mean not interval save
eval_interval = 2500
grad_norm_clip_value = 5.
train_batch_size = args.batch_size
val_batch_size = train_batch_size * 4
num_training_steps = 600000
# watch_interval = num_training_steps // 100000
watch_interval = 100
device = torch.device(f'cuda:{GPU}')
model_config = default_configs[args.model_size]


def train():

    # init model and result saving path
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.mkdir(SAVE_DIR)
    logger = get_logger(SAVE_DIR, 'train.log')

    wandb.init(
        # set the wandb project where this run will be logged
        project=MISSION_TAG,
        dir=SAVE_DIR,
        # track hyperparameters and run metadata
        config={
            "learning_rate": LR,
            "dataset": DATASET_NAME,
            "num_training_steps": num_training_steps,
        }
    )

    # Load Data
    T = TimeTracker()
    train_set, val_set = get_dataset_from_named(DATASET_NAME, dequantilize_type=DEQUAN_TYPE, load_to_device=device)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        # generator=torch.Generator(device=device)
        # pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        # generator=torch.Generator(device=device)
        # pin_memory=True
    )
    T.report_interval_time_ms('Load data loader')
    epochs = num_training_steps // len(train_loader) + 1
    logger.info(f'total training steps:{num_training_steps} total epochs:{epochs}')

    model_config['num_features'] = train_set[0].shape[0]
    transform = create_transform(model_config)
    distribution = distributions.StandardNormal(
        (model_config['num_features'],))
    model = flows.Flow(transform, distribution).to(device).train()
    logger.info(
        f'num paramaters of {MODEL_TAG}:{get_model_size_mb(model):.5f} MB')

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0)
    T.report_interval_time_ms('create & setup model')

    # ====================================train===========================================
    best_val_score = -1e10
    global_step = 0
    epoch = 0
    nclos = 102
    model.train()
    st = time()
    while global_step < num_training_steps:
        pbar = tqdm(total=len(train_loader), ncols=nclos)
        epoch += 1
        for train_batch in train_loader:
            global_step += 1
            optimizer.zero_grad()
            log_density = model.log_prob(train_batch)
            loss = -torch.mean(log_density)  # maximize negative log likelyhood
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_norm_clip_value)
            optimizer.step()
            pbar.set_description_str(
                f'epoch:{epoch}/{epochs} step:{global_step}/{num_training_steps}({100.0 * global_step / num_training_steps:.3f}%) eta:{(num_training_steps - global_step) * (time() - st) / 3600:.3f} h loss:{loss.item():.3f}')
            st = time()
            pbar.update()

            if global_step % watch_interval == 0:
                # summer_writer.add_scalar('loss', loss.detach(), global_step)
                wandb.log({"loss": loss})
            # ====================================interval save============================================
            if save_interval != -1 and (global_step + 1) % save_interval == 0:
                model.eval()
                logger.info(
                    f'saving for step {global_step}, best val score is {best_val_score}')
                torch.save(model, os.path.join(f'step-{global_step}.pt'))
                model.train()

            # ====================================valid===========================================
            if eval_interval != -1 and (global_step + 1) % eval_interval == 1:
                model.eval()
                with torch.no_grad():
                    val_logp, n_val_items = 0, 0
                    val_bar = tqdm(
                        total=len(val_loader), desc=f'evaluating for epoch {epoch}', ncols=nclos)
                    val_prefetcher = DataPrefetcher(val_loader)
                    for val_batch in val_loader:
                        n_val_items += val_batch.shape[0]
                        val_logp += model.log_prob(val_batch).detach().sum()
                        val_bar.update()
                    val_logp /= n_val_items
                    logger.info(
                        f'epoch:{epoch} val log density is {val_logp:.4f}')
                    # summer_writer.add_scalar(
                    #     'evaluate mean log probability density', val_logp, epoch)
                    wandb.log({"eval log likelyhood": val_logp})
                # ====================================save============================================
                if val_logp > best_val_score:
                    logger.info(
                        f'New best score:{val_logp:.3f} last best score:{best_val_score:.3f} !')
                    best_val_score = val_logp
                    torch.save(model, os.path.join(SAVE_DIR, 'best.pt'))

        # ================================lr schedule==========================================
        model.train()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        logger.info("schedule lr to {}".format(lr))
        # summer_writer.add_scalar("lr", lr, epoch)
        wandb.log({'lr': lr})

    wandb.finish()

if __name__ == '__main__':
    train()
    
