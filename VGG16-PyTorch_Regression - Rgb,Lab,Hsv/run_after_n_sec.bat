echo "Wait for 1h 30m,then run code!"
timeout /t 9000
echo "Starting training"
python main.py "C:\Users\22218521\Desktop\Katlego Mbatha\Collected data (2022)\regression_data\random_sets\pad_50_data_6" -a vgg16_bn --epoch 1000 -b 32 --gpu 0 --lr 1e-4 -t 0.5 --workers 10