import torch
from models.network import GPSModel
from train.trainer import create_loader, custom_train
from simple_logger import create_simple_logger
import argparse
from utils import schedule_with_warmup

def parse_args():
    parser = argparse.ArgumentParser(description='GPS Model Training')
    
    # 数据参数
    parser.add_argument('--train_csv', type=str,default='/data/young/text2/datasets/fluorescence/raw/abs_train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default='/data/young/text2/datasets/fluorescence/raw/abs_valid.csv',
                        help='Path to validation CSV file')
    parser.add_argument('--test_csv', type=str, default='/data/young/text2/datasets/fluorescence/raw/abs_test.csv',
                        help='Path to test CSV file')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--kernel', type=lambda x: list(map(int, x.split(','))), default=None,
                        help='Kernel steps (comma-separated, e.g., "1,2,4,8,16")')
    # 模型参数
    parser.add_argument('--num_layers', type=int, default=10,
                        help='Number of GPS layers')
    parser.add_argument('--dim_out', type=int, default=1,
                        help='Output dimension')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='Dropout rate')
    parser.add_argument('--attn_dropout', type=float, default=0.5,
                        help='Attention dropout rate')
    parser.add_argument('--dim_hidden', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--max_epoch', type=int, default=300,
                        help='Maximum number of epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--log_dir', type=str, default='./results',
                        help='Directory to save logs')
    
    return parser.parse_args()
def main(args):
    # 设置默认 kernel 值
    if args.kernel is None:
        args.kernel = list(range(1, 17))

    # Load dataset
    train_loader, val_loader, test_loader = create_loader(
        train_dataset_dir=args.train_csv,
        val_dataset_dir=args.val_csv,
        test_dataset_dir=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        kernel=args.kernel
    )

    # Initialize model (使用自动获取的特征维度)
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
                                             num_warmup_steps=15,
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