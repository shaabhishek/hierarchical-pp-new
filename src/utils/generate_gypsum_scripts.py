import numpy as np

def main(data_name=None):
    # Fill it with sbatch boilerplate code except the final python command
    # Load environments if you need to
    with open("./run_hpp_boilerplate.sh", "r") as file:
        boilerplate = file.read()

    print(boilerplate)

    hiddenlayers = 2**np.arange(6,11)
    batch_sizes = [32,64,128]
    lr = 1e-3
    if data_name is None:
        data_names = ["retweet"]
    else:
        data_names = [data_name]
    filenames = []
    i = 0
    for data_name in data_names:
        for hiddenlayer in hiddenlayers:
            for batch_size in batch_sizes:
                i += 1
                print("{}\t{}\t{}".format(lr, batch_size, hiddenlayer))
                sbatch_command = "python3 main.py --max_iter=100 --model=rmtpp --data_name={} --gamma=1. --l2=1e-3 --lr={} --batch_size={} --hidden_dim={}".format(data_name, lr, batch_size, hiddenlayer)
                filename = "./sbatch_hpp_{}_{}.sh".format(data_name, i)
                filenames.append(filename)
                print(filename)
                print(sbatch_command)
                print()
                with open(filename,"w+") as file:
                    file.write(boilerplate)
                    file.write(sbatch_command)
                    file.write('\n')

    # with open("./{}_files".format(data_name),"w+") as file:
    #     for fname in filenames:
    #         file.write(fname)
    #         file.write('\n')


if __name__ == "__main__":
    # Step 1: Generate the files
    main('mimic2')
    # Step 2: Run scheduler as
    # Run the `./scheduler_hpp.sh meme2`