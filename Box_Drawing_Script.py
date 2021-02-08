import numpy as np
import pandas as pd
import cv2, os, sys, random
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy import stats
from PIL import Image, ImageDraw


def contrast_stretching(input_image):
    '''
    Performing contrast stretching
    :param input_image: gray img
    :return: contrast stretched image
    '''
    # Considering 8-bit
    a, b = 0, 255
    c, d = np.amin(input_image), np.amax(input_image)
    output_image = (input_image.astype(np.float32) - c) * ((b - a) / (d - c)) + a
    return output_image.astype(np.uint8)


def new_color():
    # Red is for Mitosis, Blue is for trail
    # Yellow: [0, 211, 255]
    # Purple: [220, 100, 165]
    # Green: [0,255,0]
    # Pink: [143, 97, 228]
    # Space Cadet: [88, 55, 33]
    yellow = np.array([0,211,255])
    purple = np.array([220,100,165])
    green = np.array([0,255,0])
    pink = np.array([143, 97, 228])
    space_cadet = np.array([88, 55, 33])
    color_list = [yellow, purple, green, pink, space_cadet]
    return random.choice(color_list)


def rescaled_boundary_points(x, y, w, h):
    # Drawing Boundary box
    rescaled_start_x = x - w / 2
    rescaled_start_y = y - h / 2
    rescaled_end_x = x + w / 2
    rescaled_end_y = y + h / 2
    return rescaled_start_x, rescaled_start_y, rescaled_end_x, rescaled_end_y

def tasK_3_1(centroid_list, time_frame_list):
    if len(time_frame_list) <= 1:
        velocity_list.append(0)
        return 0
    else:
        a = np.array(centroid_list[-1])
        b = np.array(centroid_list[-2])
        dist = np.linalg.norm(a - b)
        #For statistical result analysis
        velocity_list.append(dist)
    return dist


def tasK_3_2(centroid_list, time_frame_list):
    if len(time_frame_list) <= 1:
        return 0
    else:
        dist = 0
        for index, point in enumerate(centroid_list):
            if index == 0:  # Start from second point
                continue
            a = np.array(point)
            b = np.array(centroid_list[index - 1])  # previous point
            dist += np.linalg.norm(a - b)
        return dist


def tasK_3_3(centroid_list, time_frame_list):  # Simply distance between First and last point
    if len(time_frame_list) <= 1:
        return 0
    else:
        a = np.array(centroid_list[0])
        b = np.array(centroid_list[-1])
        dist = np.linalg.norm(a - b)
        return dist


def tasK_3_4(cumulative_distance, net_distance):  # Simply distance between First and last point
    if net_distance == 0:
        return 'Division Zero'
    return cumulative_distance / net_distance

def conf_int(data, interval):
  mean, standard_error_mean = np.mean(data), stats.sem(data)
  t_value = stats.t.ppf((1+interval)/2., len(data)-1)
  lower_bound = mean - standard_error_mean*t_value
  upper_bound = mean + standard_error_mean*t_value
  return (round(lower_bound,2), round(upper_bound,2))

# Define File Path
dataset_dir = input(
    "Input Dataset Directory Name (Ensure Segmented Dir contains output images, started with 1 and *.png): ")
dataset_path = "RawData/" + dataset_dir
# Find Direcotry in Dataset
seq_dir_list = []
for _ in [x[0] for x in os.walk(dataset_path)][1:]:
    if "Masks" not in _:
        seq_dir_list.append(_.split("\\")[1])

# Questions for input:
max_distance_apart = int(input("For tracking, Enter max distance between centroids in consecutive image: "))
task_3_indicator = 0
task_3_list = []
#For statistical result analysis
velocity_list = []
# Now we loop through the each sequence
for seq in seq_dir_list:
    next_seq_indicator = False
    while not next_seq_indicator:
        # Variables for each sequence
        # Define Cell master dictionary
        cell_id = 0
        cell_master_df = pd.DataFrame(
            columns=['id', 'centroid_list', 'zlist', 'current_x', 'current_y', 'current_z', 'color', 'task_3_1', 'task_3_2', 'task_3_3', 'task_3_4'])
        # Empty list for recording previous centroid
        current_image = pd.DataFrame(columns=['x', 'y', 'w', 'h', 'id'])
        previous_image = pd.DataFrame(columns=['x', 'y', 'w', 'h', 'id'])
        # Master Color Box - There should be a better way in changing color
        master_color = np.array([0, 0, 0])
        color_channel_switch_indicator = 0

        file_path = dataset_path + "/" + seq + "/"
        file_list = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        print(f"Begining processing {seq}...")

        # Defining Initial Settings
        previous_centroid_list = []
        # Define initial color box colors
        master_color = np.array([0, 0, 0])
        # Define Cell master dictionary
        cell_id = 0
        # Master Color Box - There should be a better way in changing color
        master_color = np.array([0, 0, 0])
        color_channel_switch_indicator = 0
        previous_boundary_image_path = ""

        for file in file_list:
            print(f"Processing {file}")
            # READ IMAGE-----------------------------------------------------------------------------------
            img_num = file.split(".")[0]
            # Original - define output image
            path_img = file_path + file
            img_ori = cv2.imread(path_img)
            output_img = np.copy(img_ori)
            # Segmented Image Reading (Gray-Scale)
            seg_num = str(int(file.split(".")[0].replace("t", "")) + 1)
            seg_dir = "Segmented/" + dataset_dir + "/" + seq
            path_seg = seg_dir + '/' + seg_num + '.png'
            th_seg = cv2.imread(path_seg, cv2.IMREAD_GRAYSCALE)
            z_value = int(file.split('.')[0].replace("t", ''))
            # Finding countours-----------------------------------------------------------------------------------
            if dataset_dir == "DIC-C2DH-HeLa":
                image, cell_countours, hierachy = cv2.findContours(th_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                image, cell_countours, hierachy = cv2.findContours(th_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in cell_countours:
                x, y, w, h = cv2.boundingRect(c)
                # Find and Drawing centroid
                centroid_x, centroid_y = int(x + w / 2), int(y + h / 2)
                #TEMP:
                if task_3_indicator == 1:
                    if (centroid_x, centroid_y) in task_3_list:
                        cv2.drawMarker(output_img, (centroid_x, centroid_y), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                                       markerSize=7, thickness=1, line_type=cv2.LINE_AA)
                        # append on to current centroid list - No Id at the moment
                        current_image.loc[len(current_image)] = [centroid_x, centroid_y, w, h, 'unknown']
                else:
                    cv2.drawMarker(output_img, (centroid_x, centroid_y), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                                   markerSize=7, thickness=1, line_type=cv2.LINE_AA)
                    # append on to current centroid list - No Id at the moment
                    current_image.loc[len(current_image)] = [centroid_x, centroid_y, w, h, 'unknown']
            # For Task 3:
            cell_id_current_image = []
            if previous_image.empty:  # meaning it is the first image
                for index, cell_row in current_image.iterrows():
                    # Drawing Color
                    color_box = new_color().copy()
                    start_x, start_y, end_x, end_y = rescaled_boundary_points(cell_row['x'], cell_row['y'], cell_row['w'],
                                                                              cell_row['h'])

                    cv2.rectangle(output_img, (int(start_x), int(start_y)), (int(end_x), int(end_y)),
                                  (color_box[0].item(), color_box[1].item(), color_box[2].item()), thickness=1)
                    # Storing details
                    # cell_master_df = pd.DataFrame(columns=['id', 'x-list', 'y-list', 'z-list','current_x','current_y','current_z', 'color'])
                    cell_master_df.loc[len(cell_master_df)] = [cell_id, [(cell_row['x'], cell_row['y'])], [z_value],
                                                               cell_row['x'], cell_row['y'], z_value, color_box, 0, 0, 0, 0]
                    #changing current image id
                    current_image.at[index, 'id'] = cell_id
                    # Tracking - simply covert the current_cell dataframe 'id' to list
                    # For Task 3 Purpose:
                    cell_id_current_image.append(cell_id)
                    # Update ID:
                    cell_id += 1
            else:  # meaning there is a 'previous image'
                matched_cells = []  # [(Cell_1, Cell_2)....] List of pairs using index
                divided_cells = []  # [(Cell_1, Cell_2)....] List of pairs using index
                dummy_tracker = []  # checking which one is found, for easier processing later
                # Row is Previous Image, Column is Current Image
                previous_list = np.array(list(zip(previous_image.x, previous_image.y)))
                current_list = np.array(list(zip(current_image.x, current_image.y)))
                matrix_df = pd.DataFrame(cdist(previous_list, current_list))

                # Now we find minimum distance across column and row
                for column in range(0, len(current_list)):
                    column_min = matrix_df[column].min()
                    row_index = list(matrix_df[column]).index(column_min)
                    found_row = matrix_df.iloc[row_index]
                    row_min = found_row.min()
                    if column_min < max_distance_apart:
                        if column_min == row_min:
                            # Then these 2 cells must be equal
                            matched_cells.append((column, row_index))  # (current_image_index, previous_image_index)
                        else:
                            # If it is less than the defined distance, most likely a divided cell
                            divided_cells.append((column, row_index))

                # Processing Found Cells ---------------------------------------------------------------------------------------
                # This part can be placed into a function
                # identical cells
                for cell_pair_matched in matched_cells:
                    # We need to find the cell ID using previous cell
                    previous_x_matched = previous_image.iloc[cell_pair_matched[1]]['x']
                    previous_y_matched = previous_image.iloc[cell_pair_matched[1]]['y']
                    matched_previous_cell_id = previous_image.iloc[cell_pair_matched[1]]['id']
                    # Dont know why there is a miss match - Need to think about it
                    '''
                    matched_previous_row = cell_master_df[
                            (cell_master_df['current_x'] == previous_x) & (cell_master_df['current_y'] == previous_y)]
                    matched_previous_cell_id = matched_previous_row['id'].values[0]
                    '''
                    dummy_tracker.append(cell_pair_matched[0])
                    centroid_x_matched = current_image.iloc[cell_pair_matched[0]]['x']
                    centroid_y_matched = current_image.iloc[cell_pair_matched[0]]['y']
                    w_matched = current_image.iloc[cell_pair_matched[0]]['w']
                    h_matched = current_image.iloc[cell_pair_matched[0]]['h']
                    start_x_matched, start_y_matched, end_x_matched, end_y_matched = rescaled_boundary_points(centroid_x_matched, centroid_y_matched, w_matched, h_matched)
                    # Find Color
                    color_box = cell_master_df.loc[cell_master_df['id'] == matched_previous_cell_id]['color'].tolist()[0]
                    cv2.rectangle(output_img, (int(start_x_matched), int(start_y_matched)), (int(end_x_matched), int(end_y_matched)),
                                  (color_box[0].item(), color_box[1].item(), color_box[2].item()), thickness=1)
                    row_index = cell_master_df.id[cell_master_df.id == matched_previous_cell_id].index.tolist()[0]
                    cell_master_df.iloc[row_index]['centroid_list'] += [(centroid_x_matched, centroid_y_matched)]
                    cell_master_df.iloc[row_index]['zlist'] += [z_value]
                    cell_master_df.iloc[row_index]['current_x'] = centroid_x_matched
                    cell_master_df.iloc[row_index]['current_y'] = centroid_y_matched
                    cell_master_df.iloc[row_index]['current_z'] = z_value
                    # Draw overall Path
                    for index, i in enumerate(cell_master_df.iloc[row_index]['centroid_list']):
                        if i == cell_master_df.iloc[row_index]['centroid_list'][0]:
                            continue
                        else:
                            current_centroid = i
                            previous_centroid = cell_master_df.iloc[row_index]['centroid_list'][index-1]
                            cv2.line(output_img, current_centroid,previous_centroid, (255, 0, 0), thickness=1,lineType=8)
                    #changing current image id
                    current_image.at[cell_pair_matched[0], 'id'] = matched_previous_cell_id
                    # For Task 3 Purpose:
                    cell_id_current_image.append(matched_previous_cell_id)
                # Divided Cells - need to think about the number
                for cell_pair_divided in divided_cells:
                    division_num = 0
                    dummy_tracker.append(cell_pair_divided[0])
                    # We need to find the cell ID using previous cell
                    previous_x_divided = previous_image.iloc[cell_pair_divided[1]]['x']
                    previous_y_divided = previous_image.iloc[cell_pair_divided[1]]['y']
                    matched_previous_cell_id = previous_image.iloc[cell_pair_divided[1]]['id']
                    centroid_x_divided = current_image.iloc[cell_pair_divided[0]]['x']
                    centroid_y_divided = current_image.iloc[cell_pair_divided[0]]['y']
                    w_divided = current_image.iloc[cell_pair_divided[0]]['w']
                    h_divided = current_image.iloc[cell_pair_divided[0]]['h']
                    start_x_divided, start_y_divided, end_x_divided, end_y_divided = rescaled_boundary_points(centroid_x_divided, centroid_y_divided, w_divided,
                                                                              h_divided)
                    color_box = new_color().copy()
                    cv2.line(output_img, (centroid_x_divided, centroid_y_divided), (previous_x_divided, previous_y_divided), (255, 0, 0), thickness=1,
                             lineType=8)
                    cv2.rectangle(output_img, (int(start_x_divided), int(start_y_divided)), (int(end_x_divided), int(end_y_divided)),
                                  (color_box[0].item(), color_box[1].item(), color_box[2].item()), thickness=1)
                    # Draw onto previous image:
                    mitosis_box_start_x, mitosis_box_start_y, mitosis_box_end_x, mitosis_box_end_y = rescaled_boundary_points(
                        previous_x_divided, previous_y_divided,
                        previous_image.iloc[cell_pair_divided[1]]['w'],
                        previous_image.iloc[cell_pair_divided[1]]['h'])

                    previous_output_img = cv2.imread(previous_boundary_image_path)
                    redrawn_img = np.copy(previous_output_img)
                    cv2.rectangle(redrawn_img, (int(mitosis_box_start_x), int(mitosis_box_start_y)),
                                  (int(mitosis_box_end_x), int(mitosis_box_end_y)), (0, 0, 255), thickness=2)
                    cv2.imwrite(previous_boundary_image_path, redrawn_img)
                    # Storing Details
                    new_id = str(matched_previous_cell_id) + '-' + str(division_num)
                    # check if it already existed:
                    while True:
                        if new_id in cell_master_df['id'].values:
                            division_num += 1
                            new_id = str(matched_previous_cell_id) + '-' + str(division_num)
                        else:
                            break
                    cell_master_df.loc[len(cell_master_df)] = [new_id, [(previous_x_divided, previous_y_divided),(centroid_x_divided, centroid_y_divided)], [z_value-1, z_value], centroid_x_divided,
                                                               centroid_y_divided, z_value, color_box, 0, 0, 0, 0]
                    #changing current image id
                    current_image.at[cell_pair_divided[0], 'id'] = new_id
                    # For Task 3 Purpose:
                    cell_id_current_image.append(new_id)
                #if there is presence of divided cells
                fig_relabel = plt.figure()
                #Draw on to previous 3D graph - number of dividing cell
                previous_path_image_path = previous_boundary_image_path.replace("BoundaryBox", "PathImage")
                previous_path_img = cv2.imread(previous_path_image_path)
                fig_relabel.figimage(previous_path_img)
                txt = "Total Dividing Cell = " + str(len(divided_cells))
                plt.figtext(0.5, 0.90, txt, wrap=True, horizontalalignment='center', fontsize=11)
                plt.savefig(previous_path_image_path)
                plt.close(fig_relabel)
                # New Cells
                for column in range(0, len(current_list)):
                    # Ignore matched pairs
                    if column in dummy_tracker:
                        continue
                    # Drawing Color
                    color_box = new_color().copy()
                    # Drawing Boundary box
                    centroid_x = current_image.iloc[column]['x']
                    centroid_y = current_image.iloc[column]['y']
                    w = current_image.iloc[column]['w']
                    h = current_image.iloc[column]['h']
                    start_x, start_y, end_x, end_y = rescaled_boundary_points(centroid_x, centroid_y, w, h)
                    # Drawing box:
                    cv2.rectangle(output_img, (int(start_x), int(start_y)), (int(end_x), int(end_y)),
                                  (color_box[0].item(), color_box[1].item(), color_box[2].item()), 1)
                    cell_master_df.loc[len(cell_master_df)] = [cell_id, [(centroid_x, centroid_y)], [z_value], centroid_x,
                                                               centroid_y, z_value, color_box, 0, 0, 0, 0]
                    #changing current image id
                    current_image.at[column, 'id'] = cell_id
                    # For Calculating Task 3
                    cell_id_current_image.append(cell_id)
                    # For next new cell
                    cell_id += 1

            # Drawing onto image
            output_box_image_path = 'Results-BoundaryBox/' + dataset_dir + '/' + seq + '/' + file
            cv2.imwrite(output_box_image_path, output_img)
            #Recording as previous_image_path for next image
            previous_boundary_image_path = output_box_image_path

            # First find the create another dataframe
            out_df = cell_master_df[cell_master_df['id'].isin(cell_id_current_image) == True].set_index('id')
            out_df['current_centroid'] = list(zip(out_df.current_x, out_df.current_y))
            out_df = out_df.drop(columns=['current_x', 'current_y', 'color'])
            if out_df.empty is False:
                # Task 3.1 - Speed
                out_df['task_3_1'] = out_df.apply(lambda x: tasK_3_1(x.centroid_list, x.zlist), axis=1)
                # Task 3.2 - Cumulative Distance Travelled
                out_df['task_3_2'] = out_df.apply(lambda x: tasK_3_2(x.centroid_list, x.zlist), axis=1)
                # Task 3.3 - Net Distance Travelled
                out_df['task_3_3'] = out_df.apply(lambda x: tasK_3_3(x.centroid_list, x.zlist), axis=1)
                # Task 3.4 - Cumulative Distance / Net Distance Travelled
                out_df['task_3_4'] = out_df.apply(lambda x: tasK_3_4(x.task_3_2, x.task_3_3), axis=1)
            #Updating Cell Master Dataframe
            for row in out_df.itertuples():
                row_index = cell_master_df.id[cell_master_df.id == row.Index].index.tolist()[0]
                cell_master_df.iloc[row_index]['task_3_1'] = row.task_3_1
                cell_master_df.iloc[row_index]['task_3_2'] = row.task_3_2
                cell_master_df.iloc[row_index]['task_3_3'] = row.task_3_3
                cell_master_df.iloc[row_index]['task_3_4'] = row.task_3_4
            annotation_path = 'Results-Annotation/' + dataset_dir + '/' + seq + '/' + file.split('.')[0] + '.html'
            with open(annotation_path, 'w') as outfile:
                outfile.write(out_df.to_html())
            # Plot Graph
            fig = plt.figure()
            ax = plt.gca(projection='3d')
            y_height, x_width = th_seg.shape
            plt.xlim(0, x_width)
            plt.ylim(0, y_height)
            for row in cell_master_df.itertuples(index=False):
                z = row.zlist
                x, y = map(list, zip(*row.centroid_list))
                color_b = row.color[0] / 255
                color_g = row.color[1] / 255
                color_r = row.color[2] / 255
                ax.plot(x, y, z, color=[color_r, color_g, color_b])
            # set view angles to get better plot
            ax.set_zlim(0, len(file_list))
            ax.set_xlabel('Image X axis')
            ax.set_ylabel('Image Y axis')
            ax.set_zlabel('Time Frame')
            title = file + " (Total Cell Num = " + str(len(current_image)) + ")"
            ax.set_title(title)
            pathImage_path = 'Results-PathImage/' + dataset_dir + '/' + seq + '/' + file
            if task_3_indicator == 1:
                #This is for selected cell only - Need improvement on selecting cell in above part
                text_y_level = 0.01
                for row in out_df.itertuples():
                    #Adding txt for Task 3:
                    txt = "id="+str(row.Index)+" (x,y)="+str(row.current_centroid)+" t3-1="+str(round(row.task_3_1))+" t3-2="+str(round(row.task_3_2))+" t3-3="+str(round(row.task_3_3))
                    if type(row.task_3_4) == str:
                        txt += " t3-4=" + str(row.task_3_4)
                    else:
                        txt += " t3-4=" + str(round(row.task_3_4))
                    plt.figtext(0.5, text_y_level, txt, wrap=True, horizontalalignment='center', fontsize=11)
                    text_y_level+=0.05
            fig.savefig(pathImage_path)
            plt.close(fig)

            # Resetting
            previous_image = current_image.copy()
            current_image = pd.DataFrame(columns=['x', 'y', 'w', 'h', 'id'])
        #For statistical Analysis
        print("Analysis for entire sequence:")
        print(f"Total Cell number = {len(cell_master_df)}")
        print(f"Count Divided Cells = {cell_master_df.id.str.contains(r'-').sum()}")
        print(f"Average speed across frame and cell= {np.mean(velocity_list)}")
        print(f"Speed CF = {conf_int(velocity_list, 0.90)}")
        #Clear velocity list:
        velocity_list = []
        print(f"Average Cumulative Distance = {np.mean(cell_master_df['task_3_2'].to_numpy())}")
        print(f"Cumulative Distance CF = {conf_int(cell_master_df['task_3_2'].to_numpy(), 0.90)}")
        print(f"Average Net Distance = {np.mean(cell_master_df['task_3_3'].to_numpy())}")
        print(f"Net Distance CF = {conf_int(cell_master_df['task_3_3'].to_numpy(), 0.90)}")
        numeric_ratio_column = pd.to_numeric(cell_master_df['task_3_4'], errors='coerce').dropna()
        print(f"Average Confinment Ratio = {np.mean(numeric_ratio_column)}")
        print(f"Confinement Ratio CF = {conf_int(numeric_ratio_column, 0.90)}")
        task_3_indicator = 0 #Clear task 3 indicator
        next_seq_indicator = True
        '''
        continue_indicator = input(f"Completed {seq}, If previous ran Task 3 with selected cell, need a full re-run to load cell dictionary again"
                                   f"\ncontinue to next (N) or motion_analysis (A), terminate via (Q) :")
        if continue_indicator.lower() == "q":
            print("Terminating the program.")
            sys.exit()
        elif continue_indicator.lower() == "n":
            next_seq_indicator = True
        elif continue_indicator.lower() == "a":
            task_3_indicator = 1
            input_request = input("Task 3 - Input cell_id for tracking, maximum 2 cells for display reason, separate by comma (e.g. 2, 2-0, find the id via latest version of html annotation file): ")
            ids = input_request.replace(" ", "").split(",")
            for id in ids:
                if "-" in id:
                    task_3_list += cell_master_df[cell_master_df.id==id]['centroid_list'].values[0]
                else:
                    task_3_list += cell_master_df[cell_master_df.id == int(id)]['centroid_list'].values[0]
        '''

