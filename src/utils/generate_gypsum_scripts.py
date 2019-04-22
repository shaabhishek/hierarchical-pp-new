import numpy as np
import os

def main(data_name=None):
    # Fill it with sbatch boilerplate code except the final python command
    # Load environments if you need to

    with open("utils/run_hpp_boilerplate.sh", "r") as file:
        boilerplate = file.read()

    print(boilerplate)

    hiddenlayers = [128, 256, 512]
    lrs = [1e-3, 1e-4, 1e-5]
    dropouts = [0.1, 0.33, 0.5]
    n_clusters = [8, 16, 32]
    time_loss = "normal"
    if data_name is None:
        data_names = ["retweet", "meme", "mimic2", "book_order", "so", "lastfm"]
    else:
        data_names = [data_name]
    filenames = []
    model = "model11"
    
    
    gpu_partitions = {
        "book_order": "m40-short",
        "so": "m40-short",
        "lastfm": "m40-short",
        "retweet": "titanx-short",
        "meme": "titanx-short",
        "mimic2": "titanx-short",
        }
    
    
    
    
    
    if not os.path.isdir('experiments'):
        os.makedirs('experiments')
    for data_name in data_names:
        if not os.path.isdir(os.path.join('experiments', data_name)):
            os.makedirs(os.path.join('experiments', data_name))
        i = 0
        
        for hiddenlayer in hiddenlayers:
            for lr in lrs:
                for dropout in dropouts:
                    for n_cluster in n_clusters:
                        i += 1
                        print("{}\t{}\t{}\t{}\t{}\t{}".format(i, data_name, hiddenlayer, lr, dropout, n_cluster))
                        sbatch_command = "python3 main.py --max_iter=100 --model={} --data_name={} --hidden_dim={} --lr={} --time_loss={} --gamma=1.".format(model, data_name, hiddenlayer, lr, time_loss)
                        filename = "./sbatch_hpp_{}_{}.sh".format(data_name, i)
                        filenames.append(filename)
                        print(filename)
                        print(sbatch_command)
                        print()
                        with open(os.path.join('experiments', data_name, filename),"w+") as file:
                            # SBATCH commands
                            file.write('#!/bin/bash')
                            file.write('\n')
                            file.write('#SBATCH --job-name=hpp_{}_{}'.format(data_name, i))
                            file.write('\n')
                            file.write('#SBATCH --partition {}'.format(gpu_partitions[data_name]))
                            file.write('\n')
                            file.write('#SBATCH --gres gpu:1')
                            file.write('\n')
                            file.write('#SBATCH --output=log_{}_{}.log'.format(data_name, i))
                            file.write('\n')
                            
                            # Load environment/module
                            file.write('. /home/abhishekshar/anaconda3/etc/profile.d/conda.sh')
                            file.write('\n')
                            file.write('conda activate hpp')
                            file.write('\n')
                            file.write('cd /home/abhishekshar/hierarchichal_point_process/src')
                            file.write('\n')

                            # Actual sbatch command
                            file.write(sbatch_command)
                            file.write('\n')

    # with open("./{}_files".format(data_name),"w+") as file:
    #     for fname in filenames:
    #         file.write(fname)
    #         file.write('\n')


if __name__ == "__main__":
    # Step 1: Generate the files
    main()
    # Step 2: Run scheduler as
    # Run the `./scheduler_hpp.sh meme2`