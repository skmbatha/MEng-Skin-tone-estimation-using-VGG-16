@echo off
setlocal enabledelayedexpansion

REM Define the base directory for your experiment
set "base_dir=Experiment\new"

REM Create a date string for the folders
for /l %%i in (1, 1, 6) do (
    set "iteration=%%i"
    
    REM Get the current date and time
    for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (
        set "day=%%a"
        set "month=%%b"
        set "year=%%c"
    )
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do (
        set "hour=%%a"
        set "minute=%%b"
    )
    set "datetime= - !day!-!month!-!year! !hour!h!minute!m"
    
    REM Create the directory with the format "<iteration count> <current time>"
    mkdir "%base_dir%\!iteration! !datetime!"
    
    REM Run different commands based on the iteration number
    if %%i==1 (
        echo Running commands for iteration 1
        python main.py "C:\Users\22218521\Desktop\Katlego Mbatha\Collected data (2022)\regression_data\regr_separate_identities\dataset_1" -a vgg16_bn --epoch 50 -b 32 --gpu 0 --lr 1e-5 --wd 1e-2 --dp 0.2 -t 0.5  --pack-9d-v1
    ) else if %%i==2 (
        echo Running commands for iteration 2
        timeout /t 60
        python main.py "C:\Users\22218521\Desktop\Katlego Mbatha\Collected data (2022)\regression_data\regr_separate_identities\dataset_1" -a vgg16_bn --epoch 50 -b 32 --gpu 0 --lr 1e-5 --wd 1e-1 --dp 0.2 -t 0.5  --pack-9d-v1
    ) else if %%i==3 (
        echo Running commands for iteration 3
        timeout /t 60
        python main.py "C:\Users\22218521\Desktop\Katlego Mbatha\Collected data (2022)\regression_data\regr_separate_identities\dataset_1" -a vgg11_bn --epoch 50 -b 32 --gpu 0 --lr 1e-5 --wd 1e-2 --dp 0.2 -t 0.5  --pack-9d-v1
    ) else if %%i==4 (
        echo Running commands for iteration 4
        timeout /t 60
        python main.py "C:\Users\22218521\Desktop\Katlego Mbatha\Collected data (2022)\regression_data\regr_separate_identities\dataset_1" -a vgg11_bn --epoch 50 -b 32 --gpu 0 --lr 1e-5 --wd 1e-1 --dp 0.2 -t 0.5  --pack-9d-v1
    ) else if %%i==5 (
        echo Running commands for iteration 5
        timeout /t 60
        python main.py "C:\Users\22218521\Desktop\Katlego Mbatha\Collected data (2022)\regression_data\regr_separate_identities\dataset_1" -a vgg16_bn --epoch 50 -b 64 --gpu 0 --lr 1e-5 --wd 1e-2 --dp 0.2 -t 0.5  --pack-9d-v1
    ) else if %%i==6 (
        timeout /t 60
        python main.py "C:\Users\22218521\Desktop\Katlego Mbatha\Collected data (2022)\regression_data\regr_separate_identities\dataset_1" -a vgg16_bn --epoch 50 -b 128 --gpu 0 --lr 1e-5 --wd 1e-2 --dp 0.2 -t 0.5  --pack-9d-v1
    )
    
    copy "model_best.pth.tar" "%base_dir%\!iteration! !datetime!"
    copy "training_logs.txt" "%base_dir%\!iteration! !datetime!"
    copy "validation_output.csv" "%base_dir%\!iteration! !datetime!"
)

endlocal
