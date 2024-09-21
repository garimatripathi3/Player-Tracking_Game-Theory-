import cv2
from tqdm import tqdm
import numpy as np
import os

from players_tracking.utils import get_players_inside_court

def get_players_mean_code_values(
                                all_frames_with_req_num_of_players,
                                video_variables,
                                match_variables,
                                global_const, teams):
    
    all_color_values = {'top': [], 'bottom': []}
        
    ## now will iterate over all the frames and get lab values for all the players in it
    for frame_num in tqdm(all_frames_with_req_num_of_players):

        # code = get_color_conversion_code()
        
        # (color, bbox, frame_number, player_pixel_ratio>0.1)
        color_value = get_all_players_color_code_value(frame_num,
                                            video_variables, match_variables,
                                            {team: teams[team].to_do_mask for team in teams},
                                            global_const)
        
        
        all_color_values["top"].extend(color_value['top'])
        all_color_values["bottom"].extend(color_value['bottom'])
    
    
    return all_color_values

import cv2
import numpy as np
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Model
from keras.layers import GlobalAveragePooling2D


def get_all_players_color_code_value(frame_number,
                                     video_variables,
                                     match_variables, 
                                     to_do_bkg_sub_for_all_frames, 
                                     global_const):
    '''
    Function to get the normalized LAB, HSV, and RGB values of the players' shirts and concatenate them.
    '''
    
    # Set the frame number to read
    video_variables.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    _, frame = video_variables.video_capture.read()
    
    # Get the YOLO bounding boxes for the current frame
    YOLO_BOXES_FOR_CURR_FRAME = match_variables.yolo_data.get(str(frame_number), [])
    top_bbox, bottom_bbox = get_players_inside_court(YOLO_BOXES_FOR_CURR_FRAME, global_const)

    # Initialize result dictionary
    color_values_res = {'top': [], 'bottom': []}
    
    bkg_color = match_variables.bkg_color_range   
    bkg_l_start, bkg_l_end = bkg_color[0]
    bkg_a_start, bkg_a_end = bkg_color[1]
    bkg_b_start, bkg_b_end = bkg_color[2]
    
    for side in ['top', 'bottom']:
        boxes = top_bbox if side == 'top' else bottom_bbox
        color_values = []
        to_do_bkg_sub = to_do_bkg_sub_for_all_frames[side]
        
        for player_idx in range(global_const["NUM_PLAYERS_PER_TEAM"]): 
            if len(boxes) > player_idx:
                x1, y1, x2, y2, _ = boxes[player_idx]
                
                # Define the shirt ROI for upper or lower body
                if side == "top":
                    shirt_roi = frame[y1:y1 + (y2 - y1) // 2, x1:x2]
                else:
                    shirt_roi = frame[y1 + (y2 - y1) // 2:y2, x1:x2]

                # Convert shirt ROI to LAB, HSV, and RGB color spaces
                shirt_roi_lab_color = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2LAB)
                shirt_roi_hsv_color = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2HSV)
                shirt_roi_rgb_color = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2RGB)
                
                # Normalize each color space by mean and standard deviation
                def normalize_color_space(color_space):
                    mean = np.mean(color_space, axis=(0, 1))
                    std = np.std(color_space, axis=(0, 1))
                    return (color_space - mean) / std
                
                # Apply normalization to LAB, HSV, and RGB
                norm_lab = normalize_color_space(shirt_roi_lab_color)
                norm_hsv = normalize_color_space(shirt_roi_hsv_color)
                norm_rgb = normalize_color_space(shirt_roi_rgb_color)

                # Stack LAB, HSV, and RGB channels into a single array for both the shirt ROI
                stacked_shirt_roi = np.concatenate((norm_lab, norm_hsv, norm_rgb), axis=2)

                # Define the background mask using the stacked shirt ROI
                stacked_shirt_roi = np.mean(stacked_shirt_roi.reshape(shirt_roi.shape[0], shirt_roi.shape[1], 3, 3), axis=3)

                # Create the background mask based on LAB ranges
                bkg_mask_l = (stacked_shirt_roi[:, :, 0] > bkg_l_start) & (stacked_shirt_roi[:, :, 0] < bkg_l_end)
                bkg_mask_a = (stacked_shirt_roi[:, :, 1] > bkg_a_start) & (stacked_shirt_roi[:, :, 1] < bkg_a_end)
                bkg_mask_b = (stacked_shirt_roi[:, :, 2] > bkg_b_start) & (stacked_shirt_roi[:, :, 2] < bkg_b_end)
                bkg_mask = bkg_mask_l & bkg_mask_a & bkg_mask_b
                
                # Create a mask for player pixels (player pixels = 1, background pixels = 0)
                player_mask = np.ones_like(bkg_mask, dtype=bool) ^ bkg_mask

                # Extract player color values based on the mask
                if to_do_bkg_sub:
                    filtered_color_values = stacked_shirt_roi[player_mask]
                else:
                    filtered_color_values = stacked_shirt_roi[np.ones_like(player_mask, dtype=bool)]
                         
                # Calculate the mean color of the player's shirt ROI
                if filtered_color_values.size > 0:
                    mean_color = np.mean(filtered_color_values, axis=0)
                else:
                    mean_color = np.array([0, 0, 0])
                
                color_values.append((mean_color, [x1, y1, x2, y2], frame_number, player_mask.sum() > 0.1 * player_mask.size))

            else:
                print(f"Player {player_idx} not found on {side} side")
        
        # Store the calculated color values for top or bottom players
        if len(color_values) == global_const["NUM_PLAYERS_PER_TEAM"]:
            color_values_res[side] = color_values
        
    return color_values_res

# def get_all_players_color_code_value(frame_number,
#                                      video_variables,
#                                      match_variables, 
#                                      to_do_bkg_sub_for_all_frames, 
#                                      global_const,
#                                      code):
#     '''
#     Function to get the lab values of the players' shirts.

#     '''
    
#     video_variables.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
#     _, frame = video_variables.video_capture.read()
    
#     YOLO_BOXES_FOR_CURR_FRAME = match_variables.yolo_data.get(str(frame_number), [])
#     top_bbox, bottom_bbox = get_players_inside_court(YOLO_BOXES_FOR_CURR_FRAME, global_const)

#     color_values_res = {'top': [], 'bottom': []}
    
#     bkg_color = match_variables.bkg_color_range   
#     bkg_l_start,bkg_l_end = bkg_color[0]
#     bkg_a_start,bkg_a_end = bkg_color[1]
#     bkg_b_start,bkg_b_end = bkg_color[2]
    
    
#     for side in ['top', 'bottom']:
#         boxes = top_bbox if side == 'top' else bottom_bbox
#         color_values = []
#         to_do_bkg_sub = to_do_bkg_sub_for_all_frames[side]
        

#         for player_idx in range(global_const["NUM_PLAYERS_PER_TEAM"]): 
            
#             if len(boxes) > player_idx:
#                 x1, y1, x2, y2,_ = boxes[player_idx]
                
#                 if side == "top":
#                     shirt_roi = frame[y1:y1 + (y2 - y1) // 2, x1:x2]
#                 else:
#                     shirt_roi = frame[y1 + (y2 - y1) // 2:y2, x1:x2]
                    
 
#                 shirt_roi_lab_color = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2LAB)
#                 frame_roi_lab_color = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                
#                 shirt_roi_req_color = cv2.cvtColor(shirt_roi, code)
                
#                 bkg_mask_l = (shirt_roi_lab_color[:, :, 0] > bkg_l_start) & (shirt_roi_lab_color[:, :, 0] < bkg_l_end)
#                 bkg_mask_a = (shirt_roi_lab_color[:, :, 1] > bkg_a_start) & (shirt_roi_lab_color[:, :, 1] < bkg_a_end)
#                 bkg_mask_b = (shirt_roi_lab_color[:, :, 2] > bkg_b_start) & (shirt_roi_lab_color[:, :, 2] < bkg_b_end)
#                 bkg_mask = bkg_mask_l & bkg_mask_a & bkg_mask_b
                
#                 frame_bkg_mask_l = (frame_roi_lab_color[:, :, 0] > bkg_l_start) & (frame_roi_lab_color[:, :, 0] < bkg_l_end)
#                 frame_bkg_mask_a = (frame_roi_lab_color[:, :, 1] > bkg_a_start) & (frame_roi_lab_color[:, :, 1] < bkg_a_end)
#                 frame_bkg_mask_b = (frame_roi_lab_color[:, :, 2] > bkg_b_start) & (frame_roi_lab_color[:, :, 2] < bkg_b_end)
#                 frame_bkg_mask = frame_bkg_mask_l & frame_bkg_mask_a & frame_bkg_mask_b
                
#                 player_mask = np.ones_like(bkg_mask, dtype=bool) ^ bkg_mask ## player pixels will be one
                
#                 mask_image = (player_mask * 255).astype(np.uint8)

#                 #num of pixels with 0
#                 player_pixel_ratio = np.count_nonzero(bkg_mask == 0) / bkg_mask.size
                
#                 # print(f"Player ratio: {player_pixel_ratio}")
                
#                 # Apply the mask to get the LAB values of interest
#                 if to_do_bkg_sub:
#                     filtered_color_values = shirt_roi_req_color[player_mask]
#                 else:
#                     filtered_color_values = shirt_roi_req_color[np.ones_like(player_mask, dtype=bool)]
                         
                    
#                 # Calculate the mean LAB value of the filtered pixels
#                 if filtered_color_values.size > 0:
#                     mean_color = np.mean(filtered_color_values, axis=0) ## black region needs not to be considered
#                 else:
#                     mean_color = np.array([0, 0, 0])
                
#                 color_values.append((mean_color, [x1, y1, x2, y2], frame_number, player_pixel_ratio>0.1))
            
#             else:
#                 print(f"Player {player_idx} not found on {side} side")
        
#         if len(color_values) == global_const["NUM_PLAYERS_PER_TEAM"]:
#             color_values_res[side] = color_values
        
#     return color_values_res
          
def get_player_without_bkg(match_variables,shirt_roi):
    bkg_color = match_variables.bkg_color_range   
    bkg_l_start,bkg_l_end = bkg_color[0]
    bkg_a_start,bkg_a_end = bkg_color[1]
    bkg_b_start,bkg_b_end = bkg_color[2]

    shirt_roi_lab_color = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2LAB)
    bkg_mask_l = (shirt_roi_lab_color[:, :, 0] > bkg_l_start) & (shirt_roi_lab_color[:, :, 0] < bkg_l_end)
    bkg_mask_a = (shirt_roi_lab_color[:, :, 1] > bkg_a_start) & (shirt_roi_lab_color[:, :, 1] < bkg_a_end)
    bkg_mask_b = (shirt_roi_lab_color[:, :, 2] > bkg_b_start) & (shirt_roi_lab_color[:, :, 2] < bkg_b_end)
    bkg_mask = bkg_mask_l & bkg_mask_a & bkg_mask_b
    
    player_mask = np.ones_like(bkg_mask, dtype=bool) ^ bkg_mask ## player pixels will be one
    
    player_mask_img = (player_mask * 255).astype(np.uint8)
    
    return player_mask_img
        

# def get_color_of_player(frame,bbox):
#     '''
#     Function to get the color of the player's shirt.
#     '''
#     x1, y1, x2, y2 = bbox
#     shirt_roi = frame[y1:y2, x1:x2]
#     shirt_roi_req_color = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2HSV)
#     return np.mean(shirt_roi_req_color, axis=(0, 1))
def get_color_of_player(frame, bbox):
    '''
    Function to get the combined color features of the player's shirt 
    from LAB, HSV, and RGB color spaces, including histograms.
    '''
    x1, y1, x2, y2 = bbox
    
    # Extract the player's shirt region from the frame
    shirt_roi = frame[y1:y2, x1:x2]
    
    # Convert to LAB color space and calculate the histogram
    shirt_roi_lab = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2LAB)
    hist_lab = cv2.calcHist([shirt_roi_lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    
    # Convert to HSV color space and calculate the histogram
    shirt_roi_hsv = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([shirt_roi_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    
    # Convert to RGB color space and calculate the histogram
    shirt_roi_rgb = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2RGB)
    hist_rgb = cv2.calcHist([shirt_roi_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    
    # Combine all the histograms into a single feature vector
    combined_color_features = np.concatenate([hist_lab, hist_hsv, hist_rgb])
    
    return combined_color_features

