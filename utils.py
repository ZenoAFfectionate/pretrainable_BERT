import tqdm
import torch
import shutil
import logging
from pathlib import Path

import torch.nn as nn
from torch.cuda import amp
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from model import BERTLM, BERT


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(handler)


def get_warmup_linear_schedule(optimizer, warmup_steps, total_steps):
    """Learning rate scheduler with linear warmup"""
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda)


class BERTTrainer:
    """ BERTTrainer make the pretrained BERT model with two LM training method """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, 
                 warmup_steps=10000, total_steps=1000000,
                 grad_clip: float = 1.0, fp16: bool = True,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device(f"cuda:{cuda_devices[0]}" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, vocab_size).to(self.device)

        self.fp16 = fp16
        self.grad_clip = grad_clip

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        # Setting the Adam optimizer with hyper-param
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=betas,
            eps=1e-8
        )
        # Setting the WL learning rate scheduler
        self.scheduler = get_warmup_linear_schedule(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )

        self.scaler = amp.GradScaler(enabled=fp16)  # mix precision training

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.mlm_loss = nn.CrossEntropyLoss(ignore_index=0)  # nn.NLLLoss(ignore_index=0)
        self.nsp_loss = nn.CrossEntropyLoss()                # nn.NLLLoss()

        self.log_freq = log_freq
        self.step_count = 0

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total Parameters: {total_params / 1e6:.2f}M")
        logger.info(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")

    def train(self, epoch):
        ''' train an epoch '''
        self.model.train()
        loss, mlm_acc, nsp_acc = self.iteration(epoch, self.train_data, train=True)
        return loss, mlm_acc, nsp_acc


    def test(self, epoch):
        ''' evaluate on test set '''
        self.model.eval()
        loss, mlm_acc, nsp_acc = self.iteration(epoch, self.test_data, train=False)
        return loss, mlm_acc, nsp_acc


    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        total_loss = 0.0
        # statistic for NSP task
        total_nsp_loss = 0.0
        total_nsp_correct = 0
        total_nsp_element = 0

        # statistic for MLM task
        total_mlm_loss = 0.0
        total_mlm_correct = 0
        total_mlm_element = 0

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc=f"{str_code} Epoch {epoch:03d}",
            total=len(data_loader),
            bar_format="{l_bar}{bar:30}{r_bar}",
            dynamic_ncols=True
        )

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            with amp.autocast(enabled=self.fp16):
                # 1. forward the NSP(next sentence prediction) and MLM(mask language modeling) training
                next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

                nsp_loss = self.nsp_loss(next_sent_output, data["is_next"])

                mlm_loss = self.mlm_loss(mask_lm_output.transpose(1, 2), data["bert_label"])

                loss = nsp_loss + mlm_loss

            # 3. backward and optimization only in train
            if train:
                # gradient scaling
                self.scaler.scale(loss).backward()

                # undo gradient scale and clip
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                # update parameters
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.step_count += 1

            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item()
            total_nsp_loss += nsp_loss.item()

            # next sentence prediction accuracy
            nsp_preds = next_sent_output.argmax(dim=-1)
            total_nsp_correct += nsp_preds.eq(data["is_next"]).sum().item()
            total_nsp_element += data["is_next"].numel()

            # masked language modeling prediction accuracy
            mlm_mask = data["bert_label"] != 0
            mlm_preds = mask_lm_output.argmax(dim=-1)
            mlm_correct = mlm_preds.eq(data["bert_label"]) & mlm_mask
            total_mlm_correct += mlm_correct.sum().item()
            total_mlm_element += mlm_mask.sum().item()

            # update progess bar
            if i % self.log_freq == 0 or i == len(data_loader) - 1:
                avg_loss = total_loss / (i + 1)
                mlm_acc = total_mlm_correct / max(1, total_mlm_element) * 100
                nsp_acc = total_nsp_correct / max(1, total_nsp_element) * 100
                current_lr = self.scheduler.get_last_lr()[0]
                
                data_iter.set_postfix({
                    "LR": f"{current_lr:.2e}",
                    "Loss": f"{avg_loss:.4f}",
                    "MLM Acc": f"{mlm_acc:.2f}%",
                    "NSP Acc": f"{nsp_acc:.2f}%"
                })

        # calculate average criterion
        avg_total_loss = total_loss / len(data_loader)
        avg_mlm_loss = total_mlm_loss / len(data_loader)
        avg_nsp_loss = total_nsp_loss / len(data_loader)
        mlm_acc = total_mlm_correct / max(1, total_mlm_element) * 100
        nsp_acc = total_nsp_correct / max(1, total_mlm_element) * 100
        
        logger.info(
            f"{str_code} Epoch {epoch} | "
            f"Avg Loss: {avg_total_loss:.4f} | "
            f"MLM Loss: {avg_mlm_loss:.4f} | "
            f"NSP Loss: {avg_nsp_loss:.4f} | "
            f"MLM Acc: {mlm_acc:.2f}% | "
            f"NSP Acc: {nsp_acc:.2f}%"
        )
        
        return avg_total_loss, mlm_acc, nsp_acc


    def save(self, epoch, store_path, is_best=False):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        Path(store_path).mkdir(parents=True, exist_ok=True)

        # consider both single GPU and multi GPU model
        model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        
        # checkpoint content
        checkpoint = {
            'epoch': epoch,
            'step': self.step_count,
            'model_state': model_state,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'config': {
                'hidden': self.bert.hidden,
                'layers': self.bert.n_layers,
                'attn_heads': self.bert.attn_heads
            }
        }
        
        # make the store path and store
        filename = f"epoch_{epoch}.pt"
        save_path = Path(store_path) / filename
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved {filename} to {save_path}")
        
        if is_best:  # if it is the best, copy one
            best_path = Path(store_path) / "best_model.pt"
            shutil.copyfile(save_path, best_path)
        
        return save_path