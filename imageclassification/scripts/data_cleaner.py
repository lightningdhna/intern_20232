import argparse
from modules import data_cleaner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('command',choices=['clean','rename','remove-duplicated'],help="The command to run")
    parser.add_argument("--folder", type=str, default="data", help="Path to the data folder")
    parser.add_argument("--categories", type=str, nargs='+')
    
    args = parser.parse_args()
    
    args.folder.strip("'")
    
    if args.command == 'rename':
        data_cleaner.rename_images(args.folder,args.categories)
    elif args.command == 'clean':
        data_cleaner.remove_invalid_images(args.folder,args.categories)
    
    elif args.command == 'remove-duplicated':
        data_cleaner.remove_duplicate_images(args.folder,args.categories)