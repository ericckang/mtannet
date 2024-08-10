#!/bin/bash

# Step 1: Run mTAN interpolation on toy dataset (or any dataset toyset chosen for demonstration)
echo "Running mTAN interpolation on toy dataset..."
cd mTAN/src
python3 tan_interpolation.py --niters 10 --lr 0.0001 --batch-size 128 --rec-hidden 32 --latent-dim 1 --length 20 --enc mtan_rnn --dec mtan_rnn --n 1000  --gen-hidden 50 --save 1 --k-iwae 5 --std 0.01 --norm --learn-emb --kl --seed 0 --num-ref-points 20 --dataset toy
cd ../..

# Step 2: Run mTAN_PrimeNet_transition.py
echo "Running mTAN to PrimeNet transition..."
python3 mTAN_PrimeNet_transition.py

# Step 3: Run pretrain.sh
echo "Running pretrain.sh..."
cd PrimeNet
chmod +x pretrain.sh
./pretrain.sh
cd ..

# Step 4: Extract the latest pre-trained model number
PRETRAIN_MODEL=$(ls PrimeNet/models | sort -n | tail -1 | sed 's/.h5$//')
echo "Using pre-trained model: $PRETRAIN_MODEL"

# Step 5: Run fine-tuning script
echo "Running fine-tuning script..."
cd PrimeNet
python3 finetune.py --niters 2 --lr 0.0001 --batch-size 128 --rec-hidden 128 --n 8000 --quantization 0.016 --save 1 --classif --num-heads 1 --learn-emb --dataset physionet --seed 0 --task classification --pretrain_model "$PRETRAIN_MODEL" --pooling ave --dev 0

echo "Pipeline completed successfully!"