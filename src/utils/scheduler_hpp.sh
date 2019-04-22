echo $data_name
for FILE in `ls experiments/${data_name}`; do
echo experiments/${data_name}/${FILE}
sbatch experiments/${data_name}/${FILE}
sleep 1
done
