python3 train_aptai.py \
  --no-laptop \
  --no-logging \
  --huggingface_model_id='facebook/wav2vec2-large-robust' \
  --target_metric='val_mean_rmse' \
  --no-target_metric_bigger_better \
  --prefix='init' \
  --num_epochs=20 \
  --num_warmup_epochs=2 \
  --num_static_epochs=8 \
  --batch_size=5 \
  --learning_rate=1e-5 \
  --lr_decay=0.96 \
  --train_val_rate='both' \