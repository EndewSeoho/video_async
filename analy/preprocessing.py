import cv2 as cv
import math

def RotateImage(img, angle, x, y):
    image_center = tuple((x, y))
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


def faceAlignment(img, ptLE, ptRE, ptLM, ptRM):
    tImg = img.copy()

    understand = False
    if min(ptLM[1], ptRM[1]) < min(ptLE[1], ptRE[1]):
        understand = True

    margin = (ptRE[0] - ptLE[0])
    marginImg = cv.copyMakeBorder(tImg, margin, margin, margin, margin, cv.BORDER_CONSTANT, 0)
    ptLE = (ptLE[0] + margin, ptLE[1] + margin)
    ptRE = (ptRE[0] + margin, ptRE[1] + margin)
    ptLM = (ptLM[0] + margin, ptLM[1] + margin)
    ptRM = (ptRM[0] + margin, ptRM[1] + margin)

    fOriginalDistance = math.sqrt(math.pow((ptLE[0]+ptRE[0])/2 - (ptLM[0] + ptRM[0])/2, 2)
                                  + math.pow((ptLE[1]+ptRE[1])/2 - (ptLM[1] + ptRM[1])/2, 2))
    fScaleFactor = 48./fOriginalDistance

    fDiv = float((ptRE[0]-ptLE[0]))
    if fDiv == 0:
        fDiv = 0.0000000000001
    Theta_Radian = math.atan(float((ptRE[1]-ptLE[1]))/fDiv)
    Theta_Degree = Theta_Radian * 180. / 3.14159265358979323846264338327

    CE = ((ptLE[0]+ptRE[0])/2, (ptLE[1]+ptRE[1])/2)

    AlignedFace = RotateImage(marginImg, Theta_Degree, CE[0], CE[1])
    ScaleImg = cv.resize(AlignedFace, (int(float(marginImg.shape[1])*fScaleFactor), int(float(marginImg.shape[0])*fScaleFactor)))
    # CE[0] = int(float(CE[0])*fScaleFactor)
    # CE[1] = int(float(CE[1])*fScaleFactor)
    centerX = int(float(CE[0]) * fScaleFactor)
    centerY = int(float(CE[1]) * fScaleFactor)


    if not understand:
        roiImg = ScaleImg[centerY - 40:centerY + 88, centerX - 64:centerX + 64].copy()
    else:
        roiImg = ScaleImg[centerY - 88:centerY + 40, centerX - 64:centerX + 64].copy()
        roiImg = cv.flip(roiImg, 0)

    return roiImg
