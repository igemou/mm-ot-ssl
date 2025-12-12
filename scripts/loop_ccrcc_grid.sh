clips=(0.5 1.0 2.0)
ots=(0.5 1.0 2.0 4.0)
mlms=(0.5 1.0 2.0)
maes=(0.5 1.0 2.0)

for clip in "${clips[@]}"; do
  for ot in "${ots[@]}"; do
    for mlm in "${mlms[@]}"; do
      for mae in "${maes[@]}"; do
        sbatch --output=results/ccrcc_${clip}_${ot}_${mlm}_${mae}.out scripts/ccrcc_grid.sh "$clip" "$ot" "$mlm" "$mae"
      done
    done
  done
done
