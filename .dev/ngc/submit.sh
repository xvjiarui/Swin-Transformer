#python .dev/ngc/submit_ngc.py configs \
#  -f .dev/ngc/submit.txt ${@:1}
python .dev/ngc/submit_ngc.py configs \
  -f .dev/ngc/submit.txt ${@:1} \
  --gpus 16
