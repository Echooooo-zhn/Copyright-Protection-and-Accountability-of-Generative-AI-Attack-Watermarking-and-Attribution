import os
from PIL import Image

im = Image.open('2-images.jpg')
imW = 128
imH = 128
#16
#30
def main():
    imgIndex = 12
    for col in range(1):
        for row in range(11):
          
            curIMG = im.crop((row*imW, col*imH, row*imW + imW, col*imH + imH))
            curIMG.save('star/' + str(imgIndex) + '.jpg')
            imgIndex += 1

if __name__ == "__main__":
    main()