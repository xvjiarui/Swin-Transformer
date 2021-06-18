#python .dev/ngc/submit_ngc.py configs \
#  -f .dev/ngc/submit.txt ${@:1}

#python .dev/ngc/submit_ngc.py configs \
#  -f .dev/ngc/submit.txt ${@:1} \
#  --gpus 8 --mem 32 --ace-type norm.beta --wandb --keep 1 --batch-size 256

#python .dev/ngc/submit_ngc.py configs \
#  -f .dev/ngc/submit.txt ${@:1} \
#  --gpus 32 --mem 32 --wandb --keep 1 --batch-size 64 --amp-opt-level O0 --use-checkpoint

#python .dev/ngc/submit_ngc.py configs \
#  -f .dev/ngc/submit.txt ${@:1} \
#  --gpus 8 --mem 16 --ace-type norm --data-type ngc --wandb --keep 1

python .dev/ngc/submit_ngc.py configs \
  -f .dev/ngc/submit.txt ${@:1} \
  --gpus 8 --mem 32 --ace-type norm --data-type ngc --wandb --keep 1 --batch-size 256
