import glob
import argparse
import cv2
import os
from shutil import copyfile

parser = argparse.ArgumentParser(description='Convert Yolo relative to Abs cordinates')
parser.add_argument('-i', '--test_dir_path', type=str, required= True, help='Path for test images')
parser.add_argument('-o', '--res_dir_path', type=str, help='Path for result dir')
parser.add_argument('-d', '--draw', action='store_true', help= "Use this bool flag if you want to draw on the boxes pn the image")

args = parser.parse_args()

res_dir = args.res_dir_path
if not os.path.exists(res_dir):
    os.mkdir(res_dir)

for img_path in glob.glob(args.test_dir_path + "/**/*.jpg", recursive = True):
    print()
    print()

    img = cv2.imread(img_path)
    h, w, c = img.shape
    print("img res: ", (h,w,c))

    label_path = img_path.replace(".jpg", ".txt")
    print("Working on: ", label_path)

    res = res_dir + "/" + label_path.split("/")[-1]
    abs_label_file = open(res, "a+")

    draw_img_name = img_path.split("/")[-1]
    copyfile(img_path, res_dir + "/" + draw_img_name)

    with open(label_path, 'r') as label_file:
        for line in label_file.readlines():
            rel_cords = line.split(" ")
            print("Rel cords: ", rel_cords)

            # convert the rel_cords to abs cords
            cen_x = float(rel_cords[1]) * float(w)
            cen_y = float(rel_cords[2]) * float(h)

            w_abs = float(rel_cords[3]) * float(w)
            h_abs = float(rel_cords[4]) * float(h)

            x_abs = int(cen_x) - int(w_abs/2)
            y_abs = int(cen_y) - int(h_abs/2)

            abs_cords = "human" +" " + str(x_abs) + " " + str(y_abs) + " " + str(int(w_abs)) + " " + str(int(h_abs)) + "\n"
            print("Abs cords: ",abs_cords)

            abs_label_file.write(abs_cords)

            if args.draw:
                cv2.rectangle(img, (x_abs, y_abs), (x_abs + int(w_abs), y_abs + int(h_abs)),  (0, 255, 0), 3)
                draw_img_name = img_path.split("/")[-1]
                print("drawing on: ", res_dir + "/" + draw_img_name)
                cv2.imwrite(res_dir + "/" + draw_img_name, img)


print("Completed all files")





