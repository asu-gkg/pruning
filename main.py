import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nnodes", type=int)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parser.parse_args()
    nnodes = args.nnodes
    print(nnodes)
