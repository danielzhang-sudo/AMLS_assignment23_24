import argparse
from A import acode
from B import bcode

def main(args):
    task = args.task

    if task=="A":
        acode.main(args)
    elif task=="B":
        bcode.main(args)
    else:
        print("Run with the correct arguments")
        raise 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Arguments')
    parser.add_argument('--task', type=str, help='Task A or B')
    parser.add_argument('--mode', type=str, default="testing", help='training or testing mode')
    parser.add_argument('--alg', type=str, default="nb", help="algorithm to use")
    parser.add_argument('--ckpt_path', type=str, default="", help='path of weights file')
    args = parser.parse_args()

    main(args)