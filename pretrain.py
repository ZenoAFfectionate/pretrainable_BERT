import os
import logging
import warnings
import argparse
from datetime import datetime

from torch.utils.data import DataLoader

from model import BERT
from utils import BERTTrainer
from dataset import BERTDataset, WordVocab

warnings.filterwarnings("ignore")


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"train_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train():
    parser = argparse.ArgumentParser(description="BERT Pre-training with Advanced Optimization")
    
    # data arguments 
    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="Path to training dataset")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="Path to test dataset (optional)")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="Path to vocabulary file")
    
    # model arguments 
    parser.add_argument("-hs", "--hidden", type=int, default=768, help="Hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=12, help="Number of attention heads")
    
    # training arugments 
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Warmup steps for learning rate scheduler")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    
    # system arguments
    parser.add_argument("-o", "--output_dir", required=True, type=str, help="Output directory for models and logs")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers")
    parser.add_argument("--log_freq", type=int, default=10, help="Logging frequency in steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--cuda_devices", type=str, default=None, help="Comma-separated list of CUDA device IDs")

    args = parser.parse_args()

    # set cuda
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    
    # set logs
    logger = setup_logging(args.output_dir)
    logger.info("Starting BERT pre-training with the following configuration:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    # load vocabulary table
    logger.info(f"Loading vocabulary from {args.vocab_path}")
    vocab = WordVocab.load_vocab(args.vocab_path)
    logger.info(f"Vocabulary size: {len(vocab)}")


    logger.info(f"Creating training dataset from {args.train_dataset}")
    train_dataset = BERTDataset(
        args.train_dataset, 
        vocab, 
        seq_len=args.seq_len,
        on_memory=True
    )
    
    test_dataset = None
    if args.test_dataset:
        logger.info(f"Creating test dataset from {args.test_dataset}")
        test_dataset = BERTDataset(
            args.test_dataset, 
            vocab, 
            seq_len=args.seq_len,
            on_memory=True
        )


    logger.info("Creating data loaders")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            pin_memory=True
        )


    logger.info("Initializing BERT model")
    bert = BERT(
        vocab_size=len(vocab),
        hidden=args.hidden,
        n_layers=args.layers,
        attn_heads=args.attn_heads
    )

    logger.info("Creating BERT Trainer")
    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_loader, test_dataloader=test_loader,
                          lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                          cuda_devices=args.cuda_devices, log_freq=args.log_freq)

    best_loss = float('+inf')
    
    logger.info("Start Training...")
    for epoch in range(args.epochs):
        is_best = False
        loss, _, _ = trainer.train(epoch)
        if loss < best_loss:
            best_loss = loss
            is_best = True
        trainer.save(epoch, args.output_dir, is_best)

        if test_loader is not None:
            trainer.test(epoch)
    logger.info(f"Finish Training, best acc: {best_loss:.3f}")


if __name__ == '__main__':
    train()
