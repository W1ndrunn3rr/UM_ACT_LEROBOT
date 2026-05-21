EXPERIMENTS := A B C
train-all:
	@echo "EXP=\$\${EXPERIMENTS[\$\$SLURM_ARRAY_TASK_ID-1]}" > .slurm_train_all.sh
	@cat .slurm_train_all.sh
