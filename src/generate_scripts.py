import numpy
from random import shuffle
line1 = "#!/bin/bash \n#SBATCH --gres gpu:1\n#SBATCH --mem=35000\n"
line3 ='module load python3/current\ncd /mnt/nfs/scratch1/arighosh/hierarchichal_point_process/src\n' 
line2_m40_sh = '#SBATCH --partition m40-short\n'
line2_ti_sh = '#SBATCH --partition 1080ti-short\n'
line2_x_sh = '#SBATCH --partition titanx-short\n'
line2_m40_lo = '#SBATCH --partition m40-long\n'
line2_ti_lo = '#SBATCH --partition 1080ti-long\n'
line2_x_lo = '#SBATCH --partition titanx-long\n'

gpu_partitions = {
    "book_order": line2_x_lo,
    "so": line2_x_lo,
    "lastfm": line2_x_lo,
    "retweet": line2_x_lo,
    "meme": line2_x_lo,
    "mimic2": line2_ti_lo,
    }

script_partition_key = ['data_name', 'time_loss', 'model']

def write_to_file(file_name, lines, mode ='w'):
    with open(file_name, mode) as outfile:
        outfile.writelines(lines)
def main():
    params = {}
    name = 'hpp_'
    params['hidden_dim'] = [128]
    params['lr'] = [1e-3, 1e-4, 1e-5]
    params['latent_dim'] = [ 32, 64]
    params['anneal_iter'] = [20, 50]
    params['gamma'] = [0.1, 1]
    params['maxgradnorm'] = [ 1., 10., 100.]
    params['n_cluster'] = [8, 16]
    params['dropout'] = [ 0.25, 0.4]
    params['model'] = ['model11', 'model2']
    params['max_iter'] = [50]
    params['time_loss'] = ['normal']
    params['data_name'] = ['mimic2', 'so', 'lastfm', 'retweet', 'meme', 'book_order']

    keys_ = list(params.keys())
    
    inner_keys = [k for k in keys_ if k not in script_partition_key]
    inner_counts = [len(params[k]) for k in inner_keys]
    inner_values = [params[k] for k in inner_keys]
    inner_total_params = 1   
    for c in inner_counts:
        inner_total_params *= c
    print(inner_total_params, inner_keys)

    outer_keys = [k for k in keys_ if k in script_partition_key]
    outer_counts = [len(params[k]) for k in outer_keys]
    outer_values = [params[k] for k in outer_keys]
    outer_total_params = 1   
    for c in outer_counts:
        outer_total_params *= c
    print(outer_total_params, outer_keys)
    script_files = []

    for outer_idx in range(outer_total_params):
        script_name = './../scripts/'
        outer_line = ''
        i = (int) (outer_idx +0)
        for j in range(len(outer_keys)):
            p = outer_keys[j]
            pv = outer_values[j][i%outer_counts[j]]
            if p=='data_name': dataset = pv
            i = i//outer_counts[j]
            script_name = script_name + '__'+ p + '__'+str(pv)
            outer_line = outer_line + '--'+p + '  ' +str(pv)+'  '
        script_name = script_name +'.sh'
        script_files.append('sbatch '+script_name+'\n')
        logline = '#SBATCH --output=./../scripts/'+script_name[:-3]+'_rerun.log \n'

        preped_list = [line1, gpu_partitions[dataset], logline, line3]
        list_of_command = []

        for inner_idx in range(inner_total_params):
            inner_line = 'python3 main.py '
            i = (int) (inner_idx +0)
            for j in range(len(inner_keys)):
                p = inner_keys[j]
                pv = inner_values[j][i%inner_counts[j]]
                i = i//inner_counts[j]
                inner_line = inner_line + '--'+p + '  ' +str(pv)+'  '
            final_line = inner_line + outer_line + '\n'
            list_of_command.append(final_line)
        shuffle(list_of_command)
        preped_list.extend(list_of_command)
        write_to_file(script_name, [])
        write_to_file(script_name, preped_list , 'w')
            
    final_script_file = './../scripts/main_'+name+'.sh'
    write_to_file(final_script_file, script_files)



if __name__ == '__main__':
   main()