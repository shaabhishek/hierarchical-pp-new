for FILE in sbatch_hpp_${1}_*; do
echo ${FILE}
sbatch ${FILE} --output=log_${FILE}.log
sleep 1
done