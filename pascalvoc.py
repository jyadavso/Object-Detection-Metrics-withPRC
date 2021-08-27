###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: Oct 9th 2018                                                 #
###########################################################################################

import argparse
import glob
import os
import shutil
# from argparse import RawTextHelpFormatter
import sys
import  csv

from numpy import *
import math

import _init_paths
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!


from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import BBFormat

import pandas



# Validate formats
def ValidateFormats(argFormat, argName, errors):
    if argFormat == 'xywh':
        return BBFormat.XYWH
    elif argFormat == 'xyrb':
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append(
            'argument %s: invalid value. It must be either \'xywh\' or \'xyrb\'' % argName)


# Validate mandatory args
def ValidateMandatoryArgs(arg, argName, errors):
    if arg is None:
        errors.append('argument %s: required argument' % argName)
    else:
        return True


def ValidateImageSize(arg, argName, argInformed, errors):
    errorMsg = 'argument %s: required argument if %s is relative' % (argName, argInformed)
    ret = None
    if arg is None:
        errors.append(errorMsg)
    else:
        arg = arg.replace('(', '').replace(')', '')
        args = arg.split(',')
        if len(args) != 2:
            errors.append(
                '%s. It must be in the format \'width,height\' (e.g. \'600,400\')' % errorMsg)
        else:
            if not args[0].isdigit() or not args[1].isdigit():
                errors.append(
                    '%s. It must be in INdiaTEGER the format \'width,height\' (e.g. \'600,400\')' %
                    errorMsg)
            else:
                ret = (int(args[0]), int(args[1]))
    return ret


# Validate coordinate types
def ValidateCoordinatesTypes(arg, argName, errors):
    if arg == 'abs':
        return CoordinatesType.Absolute
    elif arg == 'rel':
        return CoordinatesType.Relative
    elif arg is None:
        return CoordinatesType.Absolute  # default when nothing is passed
    errors.append('argument %s: invalid value. It must be either \'rel\' or \'abs\'' % argName)


def ValidatePaths(arg, nameArg, errors):
    if arg is None:
        errors.append('argument %s: invalid directory' % nameArg)
    elif os.path.isdir(arg) is False and os.path.isdir(os.path.join(currentPath, arg)) is False:
        errors.append('argument %s: directory does not exist \'%s\'' % (nameArg, arg))
    # elif os.path.isdir(os.path.join(currentPath, arg)) is True:
    #     arg = os.path.join(currentPath, arg)
    else:
        arg = os.path.join(currentPath, arg)
    return arg


def getBoundingBoxes(directory,
                     isGT,
                     bbFormat,
                     coordType,
                     allBoundingBoxes=None,
                     allClasses=None,
                     imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            if isGT:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                x = float(splitLine[1])
                y = float(splitLine[2])
                w = float(splitLine[3])
                h = float(splitLine[4])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.GroundTruth,
                    format=bbFormat)
            else:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                confidence = float(splitLine[1])
                x = float(splitLine[2])
                y = float(splitLine[3])
                w = float(splitLine[4])
                h = float(splitLine[5])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.Detected,
                    confidence,
                    format=bbFormat)
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        fh1.close()
    return allBoundingBoxes, allClasses


# Get current path to set default folders
currentPath = os.path.dirname(os.path.abspath(__file__))

VERSION = '0.1 (beta)'

parser = argparse.ArgumentParser(
    prog='Object Detection Metrics - Pascal VOC',
    description='This project applies the most popular metrics used to evaluate object detection '
    'algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, '
    'please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics',
    epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)")
# formatter_class=RawTextHelpFormatter)
parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
# Positional arguments
# Mandatory
parser.add_argument(
    '-gt',
    '--gtfolder',
    dest='gtFolder',
    default=os.path.join(currentPath, 'groundtruths'),
    metavar='',
    help='folder containing your ground truth bounding boxes')
parser.add_argument(
    '-det',
    '--detfolder',
    dest='detFolder',
    default=os.path.join(currentPath, 'detections'),
    metavar='',
    help='folder containing your detected bounding boxes')
# Optional
parser.add_argument(
    '-t',
    '--threshold',
    dest='iouThreshold',
    type=float,
    default=0.5,
    metavar='',
    help='IOU threshold. Default 0.5')
parser.add_argument(
    '-gtformat',
    dest='gtFormat',
    metavar='',
    default='xywh',
    help='format of the coordinates of the ground truth bounding boxes: '
    '(\'xywh\': <left> <top> <width> <height>)'
    ' or (\'xyrb\': <left> <top> <right> <bottom>)')
parser.add_argument(
    '-detformat',
    dest='detFormat',
    metavar='',
    default='xywh',
    help='format of the coordinates of the detected bounding boxes '
    '(\'xywh\': <left> <top> <width> <height>) '
    'or (\'xyrb\': <left> <top> <right> <bottom>)')
parser.add_argument(
    '-gtcoords',
    dest='gtCoordinates',
    default='abs',
    metavar='',
    help='reference of the ground truth bounding box coordinates: absolute '
    'values (\'abs\') or relative to its image size (\'rel\')')
parser.add_argument(
    '-detcoords',
    default='abs',
    dest='detCoordinates',
    metavar='',
    help='reference of the ground truth bounding box coordinates: '
    'absolute values (\'abs\') or relative to its image size (\'rel\')')
parser.add_argument(
    '-imgsize',
    dest='imgSize',
    metavar='',
    help='image size. Required if -gtcoords or -detcoords are \'rel\'')
parser.add_argument(
    '-sp', '--savepath', dest='savePath', metavar='', help='folder where the plots are saved')
parser.add_argument(
    '-np',
    '--noplot',
    dest='showPlot',
    action='store_false',
    help='no plot is shown during execution')
args = parser.parse_args()

iouThreshold = args.iouThreshold

# Arguments validation
errors = []
# Validate formats
gtFormat = ValidateFormats(args.gtFormat, '-gtformat', errors)
detFormat = ValidateFormats(args.detFormat, '-detformat', errors)
# Groundtruth folder
if ValidateMandatoryArgs(args.gtFolder, '-gt/--gtfolder', errors):
    gtFolder = ValidatePaths(args.gtFolder, '-gt/--gtfolder', errors)
else:
    # errors.pop()
    gtFolder = os.path.join(currentPath, 'groundtruths')
    if os.path.isdir(gtFolder) is False:
        errors.append('folder %s not found' % gtFolder)
# Coordinates types
gtCoordType = ValidateCoordinatesTypes(args.gtCoordinates, '-gtCoordinates', errors)
detCoordType = ValidateCoordinatesTypes(args.detCoordinates, '-detCoordinates', errors)
imgSize = (0, 0)
if gtCoordType == CoordinatesType.Relative:  # Image size is required
    imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)
if detCoordType == CoordinatesType.Relative:  # Image size is required
    imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)
# Detection folder
if ValidateMandatoryArgs(args.detFolder, '-det/--detfolder', errors):
    detFolder = ValidatePaths(args.detFolder, '-det/--detfolder', errors)
else:
    # errors.pop()
    detFolder = os.path.join(currentPath, 'detections')
    if os.path.isdir(detFolder) is False:
        errors.append('folder %s not found' % detFolder)
if args.savePath is not None:
    savePath = ValidatePaths(args.savePath, '-sp/--savepath', errors)
else:
    savePath = os.path.join(currentPath, 'results')
# Validate savePath
# If error, show error messages
if len(errors) is not 0:
    print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                [-detformat] [-save]""")
    print('Object Detection Metrics: error(s): ')
    [print(e) for e in errors]
    sys.exit()

# Create directory to save results
shutil.rmtree(savePath, ignore_errors=True)  # Clear folder
os.makedirs(savePath)
# Show plot during execution
showPlot = args.showPlot

# print('iouThreshold= %f' % iouThreshold)
# print('savePath = %s' % savePath)
# print('gtFormat = %s' % gtFormat)
# print('detFormat = %s' % detFormat)
# print('gtFolder = %s' % gtFolder)
# print('detFolder = %s' % detFolder)
# print('gtCoordType = %s' % gtCoordType)
# print('detCoordType = %s' % detCoordType)
# print('showPlot %s' % showPlot)

# Get groundtruth boxes
allBoundingBoxes, allClasses = getBoundingBoxes(
    gtFolder, True, gtFormat, gtCoordType, imgSize=imgSize)
# Get detected boxes
allBoundingBoxes, allClasses = getBoundingBoxes(
    detFolder, False, detFormat, detCoordType, allBoundingBoxes, allClasses, imgSize=imgSize)
allClasses.sort()

evaluator = Evaluator()
acc_AP = 0
validClasses = 0

# Plot Precision x Recall curve
detections = evaluator.PlotPrecisionRecallCurve(
    allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
    IOUThreshold=iouThreshold,  # IOU threshold
    method=MethodAveragePrecision.EveryPointInterpolation,
    showAP=True,  # Show Average Precision in the title of the plot
    showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
    savePath=savePath,
    showGraphic=showPlot)

f = open(os.path.join(savePath, 'results.txt'), 'w')
f.write('Object Detection Metrics\n')
f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
f.write('Average Precision (AP), Precision and Recall per class:')

# each detection is a class
for metricsPerClass in detections:

    # Get metric values per each class
    cl = metricsPerClass['class']
    ap = metricsPerClass['AP']
    precision = metricsPerClass['precision']
    recall = metricsPerClass['recall']
    totalPositives = metricsPerClass['total positives']
    total_TP = metricsPerClass['total TP']
    total_FP = metricsPerClass['total FP']

    # Sourabh
    int_pre = metricsPerClass['interpolated precision']
    int_rec = metricsPerClass['interpolated recall']
    #print("int_pre: ", int_pre)
    #print("int_rec: ", int_rec)
    ###

    # Sourabh
    confs = metricsPerClass['conf']
    #print('confs: ', confs)
    rows = zip(confs, recall, precision)

    if totalPositives > 0:
        # Sourabh
        with open('/home/jyadavso/sourabh/people_counter_lookdown/Object-Detection-Metrics/results/data.csv', "w") as f_csv:
            writer = csv.writer(f_csv)
            for row in rows:
                writer.writerow(row)

        # Sourabh
        """
        # Plot 3D Graph to show case relationship b/w conf vs P vs R
        ax = plt.axes(projection='3d')
        # Data for a three-dimensional line
        #zline = confs
        xline = confs
        yline = int_rec[0:len(int_rec) - 1]
        ax.plot3D(xline, yline, 'gray')
        ax.set_xlabel('$conf$', fontsize=10)
        ax.set_ylabel('$recall$', fontsize=10)
        #ax.set_zlabel('$conf$', fontsize=10)
        plt.show()
        """


        t = np.asarray(confs, dtype=np.float32)
        a = np.asarray(int_pre[0:len(int_pre) - 1], dtype=np.float32)
        b = np.asarray(int_rec[0:len(int_rec) - 1], dtype=np.float32)
        #print("t type: ", type(t), " a.type: ", type(int_pre), " npa ", type(npa))
        
        plt.plot(t, a, '-b', label='Precision')  # plotting t, a separately
        plt.plot(t, b, '-r', label='Recal')  # plotting t, b separately
        #plt.plot(t, c, 'g')  # plotting t, c separately

        plt.xlabel('conf')
        plt.legend()
        #plt.show()
        plt.savefig('/home/jyadavso/sourabh/people_counter_lookdown/Object-Detection-Metrics/results/p_r_c_analysis.png')
        ###


        # Plot mAP Comparision for 5.3 vs 6.7 based on different IoU thresholds

        """
        io = pandas.read_csv(
            '/home/jyadavso/sourabh/people_counter_lookdown/Object-Detection-Metrics/sourabh_data/comparision/customer/data_mAP.csv',
            sep=",", usecols=(0, 1, 2, 3, 4, 5,6,7,8,9,10,11))  # To read 1st,2nd and 4th columns
        #print("io: ", io['R5'])

        
        R5_iou_05 = np.asarray(io['R5_iou_05'], dtype=np.float32)
        P5_iou_05 = np.asarray(io['P5_iou_05'], dtype=np.float32)

        R6_iou_05 = np.asarray(io['R6_iou_05'], dtype=np.float32)
        P6_iou_05 = np.asarray(io['P6_iou_05'], dtype=np.float32)

        R5_iou_02 = np.asarray(io['R5_iou_02'], dtype=np.float32)
        P5_iou_02 = np.asarray(io['P5_iou_02'], dtype=np.float32)

        R6_iou_02 = np.asarray(io['R6_iou_02'], dtype=np.float32)
        P6_iou_02 = np.asarray(io['P6_iou_02'], dtype=np.float32)

        R55_iou_05 = np.asarray(io['R55_iou_05'], dtype=np.float32)
        P55_iou_05 = np.asarray(io['P55_iou_05'], dtype=np.float32)

        R55_iou_02 = np.asarray(io['R55_iou_02'], dtype=np.float32)
        P55_iou_02 = np.asarray(io['P55_iou_02'], dtype=np.float32)

        plt.plot(R5_iou_05, P5_iou_05, '#F08673', label='P@iou_0.5_NVR_5.3')  # plotting t, a separately
        plt.plot(R6_iou_05, P6_iou_05, '#7173C3', label='P@ioU_0.5_Ours_6.7')  # plotting t, b separately

        plt.plot(R5_iou_02, P5_iou_02, '#E63615', label='P@iou_0.2_NVR_5.3')  # plotting t, a separately
        plt.plot(R6_iou_02, P6_iou_02, '#13169B', label='P@ioU_0.2_Ours_6.7')  # plotting t, b separately

        plt.plot(R55_iou_05, P55_iou_05, '#7FFF00', label='P@iou_0.5_NVR_5.5')  # plotting t, a separately
        plt.plot(R55_iou_02, P55_iou_02, '#32CD32', label='P@ioU_0.2_Ours_5.5')  # plotting t, b separately

        #plt.plot(t, c, 'g')  # plotting t, c separately

        plt.xlabel('Recall')
        plt.legend()
        plt.show()
        plt.savefig('/home/jyadavso/sourabh/people_counter_lookdown/Object-Detection-Metrics/results/p_r_c_analysis.png')
        ###
        
        """


	
        # Plot P-R-C Comparision for 5.3 vs 6.7
        """
        io = pandas.read_csv(
            '/home/jyadavso/sourabh/people_counter_lookdown/Object-Detection-Metrics/sourabh_data/comparision/customer/data_PRC_iou_02_with_5.5.csv',
            sep=",", usecols=(0, 1, 2, 3, 4, 5,6,7,8))  # To read 1st,2nd and 4th columns
        # print("io: ", io['R5'])

        C5_iou_05 = np.asarray(io['C5_iou_02'], dtype=np.float32)
        R5_iou_05 = np.asarray(io['R5_iou_02'], dtype=np.float32)
        P5_iou_05 = np.asarray(io['P5_iou_02'], dtype=np.float32)

        C6_iou_05 = np.asarray(io['C6_iou_02'], dtype=np.float32)
        R6_iou_05 = np.asarray(io['R6_iou_02'], dtype=np.float32)
        P6_iou_05 = np.asarray(io['P6_iou_02'], dtype=np.float32)

        C55_iou_05 = np.asarray(io['C55_iou_02'], dtype=np.float32)
        R55_iou_05 = np.asarray(io['R55_iou_02'], dtype=np.float32)
        P55_iou_05 = np.asarray(io['P55_iou_02'], dtype=np.float32)

        plt.plot(C5_iou_05, R5_iou_05, '#E63615',  label='R_NVR_5.3')  # plotting t, a separately
        plt.plot(C5_iou_05, P5_iou_05, '#E63615', dashes=(3, 3), label='P_NVR_5.3')  # plotting t, b separately

        plt.plot(C6_iou_05, R6_iou_05, '#13169B', label='R_Ours_6.7')  # plotting t, a separately
        plt.plot(C6_iou_05, P6_iou_05, '#13169B', dashes=(3, 3), label='P_Ours_6.7')  # plotting t, b separately

        plt.plot(C55_iou_05, R55_iou_05, '#7FFF00', label='R_NVR_5.5')  # plotting t, a separately
        plt.plot(C55_iou_05, P55_iou_05, '#7FFF00', dashes=(3, 3), label='P_NVR_5.5')  # plotting t, b separately

        #plt.plot(t, c, 'g')  # plotting t, c separately

        plt.xlabel('Conf')
        plt.legend()
        plt.show()
	    """


        validClasses = validClasses + 1
        acc_AP = acc_AP + ap
        prec = ['%.2f' % p for p in precision]
        rec = ['%.2f' % r for r in recall]
        ap_str = "{0:.2f}%".format(ap * 100)
        # ap_str = "{0:.4f}%".format(ap * 100)
        print('AP: %s (%s)' % (ap_str, cl))
        f.write('\n\nClass: %s' % cl)
        f.write('\nAP: %s' % ap_str)
        f.write('\nPrecision: %s' % prec)
        f.write('\nRecall: %s' % rec)

mAP = acc_AP / validClasses
mAP_str = "{0:.2f}%".format(mAP * 100)
print('mAP: %s' % mAP_str)
f.write('\n\n\nmAP: %s' % mAP_str)
