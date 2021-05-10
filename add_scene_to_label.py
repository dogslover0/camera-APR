from os import listdir,getcwd,mkdir
from os.path import isfile,join,exists
from shutil import move
import pandas as pd
import argparse
'''
Make some little change on original label file and move it to GTPose root directory
This whole script is juse some operation about dataframe, just for replacing human work.

##ONLY WORKED FOR original Cambridge###
## You need to put the cambridge dataset under dataset/Cambridge and 7scenes under the dataset/7Scenes
   For Cambridge outdoor dataset, we will use the original label file and add some little changes to fit multi-scenes
   And for indoor 7scenes, we choose to use the same version as the Cambridge's label format, so this indoor dataset
   is not of your concern.
'''
def txt_to_csv(label_file):
    '''
    this func is to replace the one space symbol with a comma in label file with format .txt and change it to .csv file
    '''
    lf = label_file
    with open(lf, 'r+') as f:
        line = f.readlines()
        for i in range(2,len(line)):
            line[i] = line[i].replace(" ", ",")
        f.close()
        with open(lf, 'w') as fw:
            fw.writelines(line)
        fw.close()

    if (lf.rsplit("/")[-1].rsplit(".")[-1] == "csv"):
        temp_csv = lf
    else:
        temp_csv = lf + ".csv"
        move(lf, temp_csv) # change it to csv to open it
    return temp_csv


def df_operation(dataframe,label_file,scene_info,split_info):
    '''
    add some scene info to label file for cambridge dataset
    '''
    df = dataframe
    lf = label_file
    COLUMN = ["scene", "split", "seq", "img_path", "t1", "t2", "t3", "q1", "q2", "q3", "q4"]
    df.insert(0,'scene', scene_info)
    df.insert(1,'split', split_info)
    df.insert(2,'seq',1)
    df.columns= COLUMN
    # in label file, the seq format of cam is seq1,seq2, and seq-01,seq-02 for 7scenes
    for i in range(len(df.index)):
        cam_or_7scenes = str(df.iloc[i,3]).split("/")[0]
        if (cam_or_7scenes == ""):
            df.iloc[i,2] = str(df.iloc[i,3].split("/")[1])
        else:
            df.iloc[i,2] = str(df.iloc[i,3].split("/")[0])

    if str(df.iloc[0,2])[3] == "-":
        df.iloc[:,3] = df.iloc[:,0] + df.iloc[:,3] 
    else:
        df.iloc[:,3] = df.iloc[:,0] + "/" + df.iloc[:,3]
    return df


root_path = getcwd() # the absolute location of this script are
original_data = join(root_path,"dataset")  # dataset with absolute full path

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("mode", help="cambridge or 7scenes")
args = arg_parser.parse_args()

mode = args.mode
# This is for Cambridge dataset
if mode == "cambridge":
    cambridge = join(original_data,"Cambridge") # dataset/Cambridge
    data = [ join(cambridge, dir) for dir in listdir(cambridge) ] # dataset/Cambridge/ShopFacade ...
    for j in range(len(data)):
        label_files = [ join(data[j],label_f) for label_f in listdir(data[j]) if isfile(join(data[j],label_f)) and label_f.split("_")[0] == "dataset" ]
        for lf in label_files:
            # call txt_to_csv func to take effect
            temp_csv = txt_to_csv(lf)
            df = pd.read_csv(temp_csv, skiprows=[0,1,2], header=None)

            # call df_operation func to add some scene info to label file
            # because this four info is different from cambridge and 7scenes, so take them in different way
            _scene = lf.rsplit("/")[-2]
            _split = lf.rsplit("/")[-1].split("_")[1].split(".")[0]

            df = df_operation(df, lf, _scene, _split)
            ### move changed label file ###
            new_label_name = lf.rsplit("/")[-2].lower() + "_" + lf.rsplit("/")[-1].split("_")[1].split(".")[0] + ".csv"
            des_data_path = lf.rsplit("/")[-3] + "_train_val" + "/" + lf.rsplit("/")[-2]
            if (not exists(join(root_path,"GTPose",des_data_path))):
                mkdir(join(root_path,"GTPose",des_data_path))
            des_path = join(root_path,"GTPose",des_data_path,new_label_name) # move processed label file to GTPose folder under root worktree
            df.to_csv(des_path,index=False)

#

# This is for 7scenes dataset, we will use available label files under GTPose 7scenes to generate new label file
else:
    seven_scenes_data = join(root_path,"dataset","7Scenes") # dataset/7Scenes/....
    seven_scenes_lf = join(seven_scenes_data,"label_before_mod") # GTPose/7Scenes_train_val
    transfer_path = join(root_path,"GTPose","7Scenes_train_val")
    # slfs stands for 7scenes label files
    slfs = [ join(seven_scenes_lf, lf) for lf in listdir(seven_scenes_lf) ] # GTPose/7Scenes_train_val/....
    for i in range(len(slfs)):
        temp_csv1 = txt_to_csv(slfs[i])
        df1 = pd.read_csv(temp_csv1, skiprows=[0,1,2], header=None)
        # four infos need to be adding into label file
        _scene1 = slfs[i].rsplit("/")[-1].split("_")[0]
        _split1 = slfs[i].rsplit("/")[-1].split("_")[1].split(".")[0]
        df1 = df_operation(df1, slfs[i],_scene1,_split1)
        new_label_name1 = slfs[i].rsplit("/")[-1].split(".")[0] + ".csv"
        train_val = slfs[i].rsplit("_")[-1].split(".")[0]
        des_path1 = join(transfer_path, train_val)
        if (not exists(des_path1)):
            mkdir(des_path1)
        print("Processing, %d done, %d remaining..."%(i,len(slfs)-i))
        des_path_final = join(des_path1, new_label_name1)
        print(des_path_final)
        df1.to_csv(des_path_final, index=False)

    
