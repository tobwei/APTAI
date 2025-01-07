python3 train_phoneme_recognizer.py \
  --no-laptop \
  --logging \
  --prefix='bestv2_w2v2robust' \
  --huggingface_model_id='facebook/wav2vec2-large-robust' \
  --cp_csv_path='../data/CommonPhone/commonphone.csv' \
  --hprc_csv_path='../data/HPRC_prep/hprc.csv' \
  --num_epochs=160 \
  --num_warmup_epochs=10 \
  --num_static_epochs=30 \
  --samples_per_epoch=2000 \
  --batch_size=2 \
  --learning_rate=5e-6 \
  --lr_decay=0.96 \
  --final_dropout=0.1 \
  --no-cropping \
  --no-ten_ms \
  --num_hidden_layers=24 \


