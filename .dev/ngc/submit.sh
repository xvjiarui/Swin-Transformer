#python .dev/ngc/submit_ngc.py configs \
#  -f .dev/ngc/submit.txt ${@:1}

#python .dev/ngc/submit_ngc.py configs \
#  -f .dev/ngc/submit.txt ${@:1} \
#  --gpus 8 --mem 32 --ace-type norm.beta --wandb --use-checkpoint

#python .dev/ngc/submit_ngc.py configs \
#  -f .dev/ngc/submit.txt ${@:1} \
#  --gpus 16 --mem 32 --wandb

python .dev/ngc/submit_ngc.py configs \
  -f .dev/ngc/submit.txt ${@:1} \
  --gpus 8 --mem 16 --ace-type norm --data-type ngc --wandb --keep 1 --use-checkpoint
