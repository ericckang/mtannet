python3 finetune.py --niters 2 --lr 0.0001 --batch-size 16 --rec-hidden 128 --n 8000 --quantization 0.016 \
--save 1 --num-heads 1 --learn-emb --dataset physionet --seed 0 --task classification \
--pretrain_model 86467 --pooling ave --dev 0