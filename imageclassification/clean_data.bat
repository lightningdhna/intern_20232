@echo off
python -m scripts.data_cleaner clean --folder "data" 
python -m scripts.data_cleaner remove-duplicated --folder "data" 
python -m scripts.data_cleaner rename --folder "data" 