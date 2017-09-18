import sys

def main():
    sys.argv.extend(["-data", "/Users/juanjose.lopez/imagine-learning/Torchevere-ML/pyTorchTraining/TTL_Sequence/model/baseline_v1"])
    import train
    train.main()

if __name__ == "__main__":
    main()