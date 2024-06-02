"""
Generate a Video (mp4) from a series of labeled img
"""
import cv2
import glob

FPS = 30

input_path = r'E:\OneDrive - Louisiana State University\000\201005_LWT_openCV\44_Python\result_230207_GOPR0539'
output_file = r'230207_GOPR0539.mp4'

img_array = []
for filename in glob.glob(input_path+r'\*.jpg'):
    img = cv2.imread(filename)
    img_array.append(img)
if not img_array:
    raise Exception("No input images")

height, width, layers = img_array[0].shape
out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), FPS, (width, height))

for i in range(len(img_array)):
    out.write(img_array[i])
    print(f'Wrote image {i + 1}/{len(img_array)}')
out.release()
print(f'Done, generated video to {output_file}')
