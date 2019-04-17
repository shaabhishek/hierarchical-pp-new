for FILE in sbatch_hpp_${1}_*; do
echo ${FILE}
sbatch ${FILE}
sleep 1
done