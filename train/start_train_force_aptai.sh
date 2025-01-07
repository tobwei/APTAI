python3 train_force_aptai.py \
  --no-laptop \
  --logging \
  --pr_model_path='../models/w2v2_phon_rec/wav2vec2-large-robust' \
  --target_metric='val_mean_rmse' \
  --no-target_metric_bigger_better \
  --prefix='Final' \
  --num_epochs=60 \
  --num_warmup_epochs=5 \
  --num_static_epochs=15 \
  --batch_size=5 \
  --learning_rate=1e-5 \
  --lr_decay=0.96 \
  --train_val_rate='N' \

