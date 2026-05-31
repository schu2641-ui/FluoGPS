import torch
from models.network import GPSModel
from train.trainer import create_loader, custom_train
from simple_logger import create_simple_logger
import argparse
from utils import schedule_with_warmup
import os

def parse_args():
    parser = argparse.ArgumentParser(description='GPS Model Training')

    # 获取当前脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, 'datasets/fluorescence/raw')

    # 数据参数
    parser.add_argument('--train_csv', type=str,
                        default=os.path.join(default_data_dir, 'e_train.csv'),
                        help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str,
                        default=os.path.join(default_data_dir, 'e_valid.csv'),
                        help='Path to validation CSV file')
    parser.add_argument('--test_csv', type=str,
                        default=os.path.join(default_data_dir, 'e_test.csv'),
                        help='Path to test CSV file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--kernel',type=list,  default=list(range(1,17)),
                        help='List of kernel sizes for RWSE')
    # 模型参数
    parser.add_argument('--num_layers', type=int, default=10,
                        help='Number of GPS layers')
    parser.add_argument('--dim_out', type=int, default=1,
                        help='Output dimension')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='Attention dropout rate')
    parser.add_argument('--dim_hidden', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')

    # 训练参数
    parser.add_argument('--lr', type=float, default=0.0007,
                        help='Learning rate')
    parser.add_argument('--warm_steps', type=float, default=6,
                        help='Number of warm-up steps')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay')
    parser.add_argument('--max_epoch', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--log_dir', type=str, default='./results',
                        help='Directory to save logs')

    return parser.parse_args()
def main(args):

    # Load dataset
    train_loader, val_loader, test_loader = create_loader(
        train_dataset_dir=args.train_csv,
        val_dataset_dir=args.val_csv,
        test_dataset_dir=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        kernel=args.kernel
    )

    # Initialize model 
    model = GPSModel(dim_h=args.dim_hidden,
                        num_heads=args.num_heads,
                        dropout=args.dropout,
                        attn_dropout=args.attn_dropout,
                        num_layers=args.num_layers,
                        dim_out=args.dim_out,
                        rwse_steps=args.kernel).to(args.device)
    # Create logger
    loggers = create_simple_logger(output_dir=args.log_dir)
    # Start training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = schedule_with_warmup(optimizer=optimizer,
                                             num_warmup_steps=args.warm_steps,
                                             num_training_steps=args.max_epoch)

    custom_train(model=model,
                loaders=[train_loader, val_loader, test_loader],
                loggers=loggers,
                optimizer=optimizer,
                scheduler=scheduler,
                max_epoch=args.max_epoch,)

if __name__ == '__main__':
    args = parse_args()
    main(args)