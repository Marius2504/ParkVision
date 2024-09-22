import numpy as np



def evaluate_results_task1(predictions_path,ground_truth_path,verbose = 0):
    
    total_correct_queries = 0

    for i in range(1,16):
        if(i<10):
            name = '0' + str(i)
        else:
            name = str(i)
            
        for j in range(1,5):
            if (i<11) and (j==4):
                continue
            
            correct_prediction = 0
            try:
                filename_predictions = predictions_path  + name + "_" + str(j) + "_predicted.txt"
                filename_ground_truth = ground_truth_path + name + "_" + str(j) + "_gt.txt"

                p = open(filename_predictions,"rt") 
                gt = open(filename_ground_truth,"rt")
               
                correct_prediction = 1

                #read the first line - number of query lanes
                p_line = p.readline()
                gt_line = gt.readline()

                nb_gt_lines = int(gt_line[:1])          
                #read the next nb_gt_lines
                for q in range(1,nb_gt_lines+1):
                    p_line = p.readline()       
                    gt_line = gt.readline()
                    #print(p_line,gt_line)

                    if (p_line[:3] != gt_line[:3]):
                        correct_prediction = 0
                        break
                
                p.close()
                gt.close()
    
            except:
                print("Error")


            if verbose:
                print("Task 1 - classifying on-street parking places: for test example number ", str(i), "image ", str(j), " the prediction is :", (1-correct_prediction) * "in" + "correct" + "\n")
               
            total_correct_queries = total_correct_queries + correct_prediction
        
    points = total_correct_queries * 0.03
    return total_correct_queries, points


def evaluate_results_task2(predictions_path,ground_truth_path,verbose = 0):
    
    total_correct_configurations = 0

    for i in range(1,16):
        if(i<10):
            name = '0' + str(i)
        else:
            name = str(i)

            
        correct_prediction = 0
        try:
            filename_predictions = predictions_path  + name  + "_predicted.txt"
            filename_ground_truth = ground_truth_path + name + "_gt.txt"

            #print(filename_predictions,filename_ground_truth)

            p = open(filename_predictions,"rt") 
            gt = open(filename_ground_truth,"rt")
               
            correct_prediction = 1

            #read the first 10 lines
            for q in range(10):
                p_line = p.readline()       
                gt_line = gt.readline()
                
                if (p_line[:1] != gt_line[:1]):
                    correct_prediction = 0
                    break
                
            p.close()
            gt.close()
    
        except:
            print("Error")


        if verbose:
            print("Task 2 - listing configurations of the 10 on-street parking places: for test example number ", str(i), " the prediction is :", (1-correct_prediction) * "in" + "correct" + "\n")
               
        total_correct_configurations = total_correct_configurations + correct_prediction
        
    points = total_correct_configurations * 0.05
    return total_correct_configurations, points

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def compute_percentage_tracking(gt_bboxes, predicted_bboxes, num_frames,verbose=0):
    """
    This function compute the percentage of detected bounding boxes based on the ground-truth bboxes and the predicted ones.
    :param gt_bboxes. The ground-truth bboxes with the format: frame_idx, x_min, y_min, x_max, y_max.
    :param predicted_bboxes. The predicted bboxes with the format: frame_idx, x_min, y_min, x_max, y_max
    :param num_frames. The total number of frames in the video.
    """
    
    num_frames = int(num_frames)
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    gt_dict = {}
    for gt_box in gt_bboxes:
        gt_dict[gt_box[0]] = gt_box[1:]
    
    pred_dict = {}
    for pred_bbox in predicted_bboxes:
        pred_dict[pred_bbox[0]] = pred_bbox[1:]
        
    for i in range(num_frames):
        if gt_dict.get(i, None) is None and pred_dict.get(i, None) is None: # the bowling bowl is not on the lane
            tn += 1 
        
        elif gt_dict.get(i, None) is not None and pred_dict.get(i, None) is None: # the bowling ball is not detected
            fn += 1
            
        elif gt_dict.get(i, None) is None and pred_dict.get(i, None) is not None: # the bowling ball is not on the lane, but it is 'detected'
            fp += 1
            
        elif gt_dict.get(i, None) is not None and pred_dict.get(i, None) is not None: # the bowling ball is on the lane and it is detected
            
            iou = bb_intersection_over_union(gt_dict[i], pred_dict[i])
            if iou >= 0.3:
                tp += 1
            else:
                fp += 1 
                         
    if verbose:
        print(f'tp = {tp}, tn = {tn}, fp = {fp},fn = {fn}')
    assert tn + fn + tp + fp == num_frames
    perc = (tp + tn) / (tp + fp + tn + fn)
    
    return perc

def evaluate_results_task3(predictions_path,ground_truth_path, verbose = 0):
    total_correct_tracked_videos = 0
    for i in range(1,16):

        correct_tracked_video = 0
    
        try:

            if(i<10):
                name = '0' + str(i)
            else:
                name = str(i)
    
            filename_predictions = predictions_path + "/" + name + "_predicted.txt"
            filename_ground_truth = ground_truth_path + "/" + name + "_gt.txt"

            p = np.loadtxt(filename_predictions)
            predicted_bboxes = p[1:]
            gt = np.loadtxt(filename_ground_truth)
            gt_bboxes = gt[1:]
            num_frames = gt[0][0]
            percentage = compute_percentage_tracking(gt_bboxes, predicted_bboxes, num_frames,verbose)

            correct_tracked_video = 1
            if percentage < 0.8:
                correct_tracked_video = 0
        
    
            if verbose:
                print("percentage = ", percentage)
                print("Task 3 - Tracking the vehicle: for test example number ", str(i), " the prediction is :", (1-correct_tracked_video) * "in" + "correct", "\n")
        
            total_correct_tracked_videos = total_correct_tracked_videos + correct_tracked_video
    
        except:
            print("Error")

        points = total_correct_tracked_videos * 0.1
        
    return total_correct_tracked_videos,points 



def evaluate_results_task4(predictions_path,ground_truth_path,verbose = 0):
    
    total_correct_predictions = 0

    for i in range(1,16):
        if(i<10):
            name = '0' + str(i)
        else:
            name = str(i)
            
        
            
        correct_prediction = 0
        try:
            filename_predictions = predictions_path  + name + "_predicted.txt"
            filename_ground_truth = ground_truth_path + name + "_gt.txt"

            p = open(filename_predictions,"rt") 
            gt = open(filename_ground_truth,"rt")
               
            correct_prediction = 1
                
            #read 1 line
            for q in range(1):
                p_line = p.readline()       
                gt_line = gt.readline()
                
                if (p_line.strip() != gt_line.strip()):
                    correct_prediction = 0
                    break
                
            p.close()
            gt.close()
    
        except:
            print("Error")


        if verbose:
                print("Task 4 - predicting number of vechicles queing: for test example number ", str(i), " the prediction is :", (1-correct_prediction) * "in" + "correct" + "\n")
               
        total_correct_predictions = total_correct_predictions + correct_prediction
        
    points = total_correct_predictions * 0.05
    return total_correct_predictions, points



#change this on your machine
predictions_path_root = "../submission_files/407_Dumitrescu_Marius/"
ground_truth_path_root = "../../train/"

#task1
verbose = 1
predictions_path = predictions_path_root + "Task1/"
ground_truth_path = ground_truth_path_root + "Task1/ground-truth/"
total_correct_queries, points_task1 = evaluate_results_task1(predictions_path,ground_truth_path,verbose)

print("Task 1 = ", points_task1)



#task2
verbose = 1
predictions_path = predictions_path_root + "Task2/"
ground_truth_path = ground_truth_path_root + "Task2/ground-truth/"
total_correct_configurations_task2,points_task2 = evaluate_results_task2(predictions_path,ground_truth_path,verbose)
print("Task 2 = ", points_task2)



#task3
verbose = 1
predictions_path = predictions_path_root + "Task3/"
ground_truth_path = ground_truth_path_root + "Task3/ground-truth/"
total_correct_tracked_videos_task3, points_task3 = evaluate_results_task3(predictions_path,ground_truth_path,verbose)
print("Task 3 = ", points_task3)


#task4
verbose = 1
predictions_path = predictions_path_root + "Task4/"
ground_truth_path = ground_truth_path_root + "Task4/ground-truth/"
total_correct_predictions_task4, points_task4 = evaluate_results_task4(predictions_path,ground_truth_path,verbose)
print("Task 4 = ", points_task4)


print("Task 1 = ", points_task1, "Task 2 = ", points_task2,"Task 3 = ", points_task3, "Task 4 = ", points_task4)



