mkdir ../files

cd ../files

pip install kaggle
kaggle competitions download -c mayo-clinic-strip-ai

unzip mayo-clinic-strip-ai.zip

python scale_images.py -d ../../files/train/ -s ../../files/resized_train/ -i 256 -p ../issue_images.pickle