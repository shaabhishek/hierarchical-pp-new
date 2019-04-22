echo $data_name
for FILE in `ls experiments/${data_name}`; do
echo experiments/${data_name}/${FILE}
sbatch ${FILE}
sleep 1
done