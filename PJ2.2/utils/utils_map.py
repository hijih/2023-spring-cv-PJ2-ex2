import glob
import json
import math
import os
import shutil
import sys
import matplotlib
import numpy as np


def log_average_miss_rate(precision, fp_cumsum, num_images):
    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    ref = np.logspace(-2.0, 0.0, num = 9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi

def error(msg):
    print(msg)
    sys.exit(0)

def voc_ap(rec, prec):

    rec.insert(0, 0.0) 
    rec.append(1.0) 
    mrec = rec[:]
    prec.insert(0, 0.0) 
    prec.append(0.0) 
    mpre = prec[:]

    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

def file_lines_to_list(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

def get_map(MINOVERLAP, draw_plot, score_threhold=0.5, path = './map_out'):
    
    GT_PATH             = os.path.join(path, 'ground-truth')
    DR_PATH             = os.path.join(path, 'detection-results')
    TEMP_FILES_PATH     = os.path.join(path, '.temp_files')
    RESULTS_FILES_PATH  = os.path.join(path, 'results')

    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)
        
    if os.path.exists(RESULTS_FILES_PATH):
        shutil.rmtree(RESULTS_FILES_PATH)
    else:
        os.makedirs(RESULTS_FILES_PATH)
    if draw_plot:
        try:
            matplotlib.use('TkAgg')
        except:
            pass
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "AP"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "F1"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Recall"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Precision"))

    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    gt_counter_per_class     = {}
    counter_images_per_class = {}

    for txt_file in ground_truth_files_list:
        file_id     = txt_file.split(".txt", 1)[0]
        file_id     = os.path.basename(os.path.normpath(file_id))
        temp_path   = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error(error_msg)
        lines_list      = file_lines_to_list(txt_file)
        bounding_boxes  = []
        is_difficult    = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except:
                if "difficult" in line:
                    line_split  = line.split()
                    _difficult  = line_split[-1]
                    bottom      = line_split[-2]
                    right       = line_split[-3]
                    top         = line_split[-4]
                    left        = line_split[-5]
                    class_name  = ""
                    for name in line_split[:-5]:
                        class_name += name + " "
                    class_name  = class_name[:-1]
                    is_difficult = True
                else:
                    line_split  = line.split()
                    bottom      = line_split[-1]
                    right       = line_split[-2]
                    top         = line_split[-3]
                    left        = line_split[-4]
                    class_name  = ""
                    for name in line_split[:-4]:
                        class_name += name + " "
                    class_name = class_name[:-1]

            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes  = list(gt_counter_per_class.keys())
    gt_classes  = sorted(gt_classes)
    n_classes   = len(gt_classes)

    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            file_id = txt_file.split(".txt",1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except:
                    line_split      = line.split()
                    bottom          = line_split[-1]
                    right           = line_split[-2]
                    top             = line_split[-3]
                    left            = line_split[-4]
                    confidence      = line_split[-5]
                    tmp_class_name  = ""
                    for name in line_split[:-5]:
                        tmp_class_name += name + " "
                    tmp_class_name  = tmp_class_name[:-1]

                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " +bottom
                    bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})

        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    with open(RESULTS_FILES_PATH + "/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}

        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))

            nd          = len(dr_data)
            tp          = [0] * nd
            fp          = [0] * nd
            score       = [0] * nd
            score_threhold_idx = 0
            for idx, detection in enumerate(dr_data):
                file_id     = detection["file_id"]
                score[idx]  = float(detection["confidence"])
                if score[idx] >= score_threhold:
                    score_threhold_idx = idx

                gt_file             = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                ground_truth_data   = json.load(open(gt_file))
                ovmax       = -1
                gt_match    = -1
                bb          = [float(x) for x in detection["bbox"].split()]
                for obj in ground_truth_data:
                    if obj["class_name"] == class_name:
                        bbgt    = [ float(x) for x in obj["bbox"].split() ]
                        bi      = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                        iw      = bi[2] - bi[0] + 1
                        ih      = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                            + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj
                    
                min_overlap = MINOVERLAP
                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            with open(gt_file, 'w') as f:
                                    f.write(json.dumps(ground_truth_data))
                        else:
                            fp[idx] = 1
                else:
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"
                

            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
                
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val

            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / np.maximum(gt_counter_per_class[class_name], 1)

            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]), 1)

            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            F1  = np.array(rec)*np.array(prec)*2 / np.where((np.array(prec)+np.array(rec))==0, 1, (np.array(prec)+np.array(rec)))

            sum_AP  += ap
            text    = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)

            if len(prec)>0:
                F1_text         = "{0:.2f}".format(F1[score_threhold_idx]) + " = " + class_name + " F1 "
                Recall_text     = "{0:.2f}%".format(rec[score_threhold_idx]*100) + " = " + class_name + " Recall "
                Precision_text  = "{0:.2f}%".format(prec[score_threhold_idx]*100) + " = " + class_name + " Precision "
            else:
                F1_text         = "0.00" + " = " + class_name + " F1 " 
                Recall_text     = "0.00%" + " = " + class_name + " Recall " 
                Precision_text  = "0.00%" + " = " + class_name + " Precision " 

            rounded_prec    = [ '%.2f' % elem for elem in prec ]
            rounded_rec     = [ '%.2f' % elem for elem in rec ]
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
            
            if len(prec)>0:
                print(text + "\t||\tscore_threhold=" + str(score_threhold) + " : " + "F1=" + "{0:.2f}".format(F1[score_threhold_idx])\
                    + " ; Recall=" + "{0:.2f}%".format(rec[score_threhold_idx]*100) + " ; Precision=" + "{0:.2f}%".format(prec[score_threhold_idx]*100))
            else:
                print(text + "\t||\tscore_threhold=" + str(score_threhold) + " : " + "F1=0.00% ; Recall=0.00% ; Precision=0.00%")
            ap_dictionary[class_name] = ap

            n_images = counter_images_per_class[class_name]
            lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
            lamr_dictionary[class_name] = lamr

        results_file.write("\n# mAP of all classes\n")
        mAP     = sum_AP / n_classes
        text    = "mAP = {0:.2f}%".format(mAP*100)
        results_file.write(text + "\n")
        print(text)

    shutil.rmtree(TEMP_FILES_PATH)

    det_counter_per_class = {}
    for txt_file in dr_files_list:
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            if class_name in det_counter_per_class:
                det_counter_per_class[class_name] += 1
            else:
                det_counter_per_class[class_name] = 1
    dr_classes = list(det_counter_per_class.keys())

    with open(RESULTS_FILES_PATH + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")


    for class_name in dr_classes:
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0

    with open(RESULTS_FILES_PATH + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = class_name + ": " + str(n_det)
            text += " (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
            results_file.write(text)
    return mAP