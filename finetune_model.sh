source $HOME/miniconda/bin/activate

conda activate geopixel

cp drive/MyDrive/Capstone/checkpoint-800.zip .

unzip -q checkpoint-800.zip -d GeoPixel/

cd GeoPixel/

python merge_lora_weights_and_save_hf_model.py \
    --version MBZUAI/GeoPixel-7B-RES \
    --weight checkpoint-800/adapter_model.bin \
    --save_path ./merged_model

bash finetune.sh