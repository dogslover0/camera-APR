1. Check the requirements.txt, get the right version of python libraries..

2. Use the add_scene_to_label.py to add scene info to label file

   NOTE: You need to put your original cambridge dataset on the dataset/cambridge(will use the original label file to generate new label file with scene info), and for microsoft 7scenes you need to compressed them insteadly.

3. Then use scripts/combine_scene.py to generate multi-scene label-file, e.g cambridge-4scenes.csv ...

4. Use main.py to train/test  on your dataset, input main.py -h to see the args you may need !

5. 

