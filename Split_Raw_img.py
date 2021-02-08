
import cv2, os
dataset_dir = input("Input Directory path containing dataset output: ")
if 'dic' in dataset_dir:
    dataset_name = "DIC-C2DH-HeLa"
elif 'fluo' in dataset_dir:
    dataset_name = "Fluo-N2DL-HeLa"
elif 'phc' in dataset_dir:
    dataset_name = "PhC-C2DL-PSC"

dir_list = []
for _ in [x[0] for x in os.walk(dataset_dir)][1:]:
    dir_list.append(_.split("\\")[1])

for dir in dir_list:
    print(f"Processing {dir}....")
    path_img_dir = dataset_dir + '/' + dir + '/'
    file_list = [f for f in os.listdir(path_img_dir) if os.path.isfile(os.path.join(path_img_dir, f))]
    for file in file_list:
        file_path = path_img_dir + file
        img = cv2.imread(file_path)
        width, height, channels = img.shape
        ori_img = img[:,:int(height/3),:]
        seg = img[:,int(height/3):int(height*2/3),:]
        dis = img[:,int(height*2/3):,:]
        #Write seg_img only, try and use give raw

        seg_out_path = 'Segmented/' + dataset_name +'(Raw)/' + dir + '/' + file
        cv2.imwrite(seg_out_path, seg)
        dis_out_path = seg_out_path.replace('(Raw)', '(DisT)')
        cv2.imwrite(dis_out_path, dis)


