import cv2
import time
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import Delaunay
from scipy.interpolate import RectBivariateSpline
from matplotlib.path import Path

class Triangle:
    def __init__(self, vertices):
        if isinstance(vertices, np.ndarray) == 0:
            raise ValueError("Input argument is not of type np.array.")
        if vertices.shape != (3, 2):
            raise ValueError("Input argument does not have the expected dimensions.")
        if vertices.dtype != np.float64:
            raise ValueError("Input argument is not of type float64.")
        self.vertices = vertices
        self.minX = int(self.vertices[:, 0].min())
        self.maxX = int(self.vertices[:, 0].max())
        self.minY = int(self.vertices[:, 1].min())
        self.maxY = int(self.vertices[:, 1].max())

    def getPoints(self):
        xList = range(self.minX, self.maxX + 1)
        yList = range(self.minY, self.maxY + 1)
        emptyList = list((x, y) for x in xList for y in yList)

        points = np.array(emptyList, np.float64)
        p = Path(self.vertices)
        grid = p.contains_points(points)
        mask = grid.reshape(self.maxX - self.minX + 1, self.maxY - self.minY + 1)

        trueArray = np.where(np.array(mask) == True)
        coordArray = np.vstack((trueArray[0] + self.minX, trueArray[1] + self.minY, np.ones(trueArray[0].shape[0])))

        return coordArray

        
class Morpher:
    def __init__(self, leftImage, leftTriangles, rightImage, rightTriangles):
        self.leftImage = np.ndarray.copy(leftImage)
        self.leftTriangles = leftTriangles
        self.rightImage = np.ndarray.copy(rightImage)
        self.rightTriangles = rightTriangles
        self.leftInterpolation = RectBivariateSpline(np.arange(self.leftImage.shape[0]), np.arange(self.leftImage.shape[1]), self.leftImage)
        self.rightInterpolation = RectBivariateSpline(np.arange(self.rightImage.shape[0]), np.arange(self.rightImage.shape[1]), self.rightImage)

    def getImageAtAlpha(self, alpha, smoothMode):
        for leftTriangle, rightTriangle in zip(self.leftTriangles, self.rightTriangles):
            self.interpolatePoints(leftTriangle, rightTriangle, alpha)

        blendARR = ((1 - alpha) * self.leftImage + alpha * self.rightImage)
        blendARR = blendARR.astype(np.uint8)
        return blendARR

    def interpolatePoints(self, leftTriangle, rightTriangle, alpha):
        targetTriangle = Triangle(leftTriangle.vertices + (rightTriangle.vertices - leftTriangle.vertices) * alpha)
        targetVertices = targetTriangle.vertices.reshape(6, 1)
        tempLeftMatrix = np.array([[leftTriangle.vertices[0][0], leftTriangle.vertices[0][1], 1, 0, 0, 0],
                                   [0, 0, 0, leftTriangle.vertices[0][0], leftTriangle.vertices[0][1], 1],
                                   [leftTriangle.vertices[1][0], leftTriangle.vertices[1][1], 1, 0, 0, 0],
                                   [0, 0, 0, leftTriangle.vertices[1][0], leftTriangle.vertices[1][1], 1],
                                   [leftTriangle.vertices[2][0], leftTriangle.vertices[2][1], 1, 0, 0, 0],
                                   [0, 0, 0, leftTriangle.vertices[2][0], leftTriangle.vertices[2][1], 1]])
        tempRightMatrix = np.array([[rightTriangle.vertices[0][0], rightTriangle.vertices[0][1], 1, 0, 0, 0],
                                    [0, 0, 0, rightTriangle.vertices[0][0], rightTriangle.vertices[0][1], 1],
                                    [rightTriangle.vertices[1][0], rightTriangle.vertices[1][1], 1, 0, 0, 0],
                                    [0, 0, 0, rightTriangle.vertices[1][0], rightTriangle.vertices[1][1], 1],
                                    [rightTriangle.vertices[2][0], rightTriangle.vertices[2][1], 1, 0, 0, 0]])

        #lefth = np.linalg.solve(tempLeftMatrix, targetVertices)
        #righth = np.linalg.solve(tempRightMatrix, targetVertices)
        lefth, _, _, _ = np.linalg.lstsq(tempLeftMatrix, targetVertices, rcond=None)
        righth, _, _, _ = np.linalg.lstsq(tempRightMatrix, targetVertices, rcond=None)

        leftH = np.array([[lefth[0][0], lefth[1][0], lefth[2][0]], [lefth[3][0], lefth[4][0], lefth[5][0]], [0, 0, 1]])
        rightH = np.array([[righth[0][0], righth[1][0], righth[2][0]], [righth[3][0], righth[4][0], righth[5][0]], [0, 0, 1]])
        leftinvH = np.linalg.inv(leftH)
        rightinvH = np.linalg.inv(rightH)
        targetPoints = targetTriangle.getPoints()

        leftSourcePoints = np.transpose(np.matmul(leftinvH, targetPoints))
        rightSourcePoints = np.transpose(np.matmul(rightinvH, targetPoints))
        targetPoints = np.transpose(targetPoints)

        for x, y, z in zip(targetPoints, leftSourcePoints, rightSourcePoints):
            self.leftImage[int(x[1])][int(x[0])] = self.leftInterpolation(y[1], y[0])
            self.rightImage[int(x[1])][int(x[0])] = self.rightInterpolation(z[1], z[0])

def loadTriangles(limg, rimg, featuregridsize) -> tuple:
    lrlists = autofeaturepoints(limg, rimg, featuregridsize)

    leftArray = np.array(lrlists[0], np.float64)
    rightArray = np.array(lrlists[1], np.float64)
    delaunayTri = Delaunay(leftArray)

    leftNP = leftArray[delaunayTri.simplices]
    rightNP = rightArray[delaunayTri.simplices]

    leftTriList = [Triangle(x) for x in leftNP]
    rightTriList = [Triangle(x) for x in rightNP]

    return leftTriList, rightTriList


def autofeaturepoints(leimg, riimg, featuregridsize):
    result = [[], []]
    for idx, img in enumerate([leimg, riimg]):
        result[idx] = [[0, 0], [(img.shape[1] - 1), 0], [0, (img.shape[0] - 1)], [(img.shape[1] - 1), (img.shape[0] - 1)]]

        h = int(img.shape[0] / featuregridsize)
        w = int(img.shape[1] / featuregridsize)
        
        for i in range(0, featuregridsize):
            for j in range(0, featuregridsize):
                crop_img = img[j * h: (j + 1) * h, i * w: (i + 1) * w]
                gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                featurepoints = cv2.goodFeaturesToTrack(gray, 1, 0.1, 10)
                if featurepoints is None:
                    featurepoints = [[[h / 2, w / 2]]]
                featurepoints = np.int32(featurepoints)
                for featurepoint in featurepoints:
                    x, y = featurepoint.ravel()
                    y = y + (j * h)
                    x = x + (i * w)
                    result[idx].append([x, y])
    return result


def morph_images(
    leftImageRaw, 
    rightImageRaw, 
    num_frames, 
    output_prefix="frame", 
    feature_grid_size=7, 
    smoothing=0, 
    scale=1.0
):
    #leftImageRaw = cv2.imread(start_image_path)
    #rightImageRaw = cv2.imread(end_image_path)
    
    if scale != 1.0:
        leftImageRaw = cv2.resize(
            leftImageRaw, 
            (int(leftImageRaw.shape[1] * scale), int(leftImageRaw.shape[0] * scale)), 
            interpolation=cv2.INTER_CUBIC
        )
        rightImageRaw = cv2.resize(
            rightImageRaw, 
            (int(rightImageRaw.shape[1] * scale), int(rightImageRaw.shape[0] * scale)), 
            interpolation=cv2.INTER_CUBIC
        )

    triangleTuple = loadTriangles(leftImageRaw, rightImageRaw, feature_grid_size)
    
    morphers = [
        Morpher(leftImageRaw[:, :, c], triangleTuple[0], rightImageRaw[:, :, c], triangleTuple[1])
        for c in range(3)
    ]

    intermediate_images = []
    for i in range(num_frames + 1):
        alpha = i / num_frames
        if smoothing > 0:
            outimage = np.dstack([
                np.array(median_filter(morphers[0].getImageAtAlpha(alpha, True), smoothing)),
                np.array(median_filter(morphers[1].getImageAtAlpha(alpha, True), smoothing)),
                np.array(median_filter(morphers[2].getImageAtAlpha(alpha, True), smoothing))
            ])
        else:
            outimage = np.dstack([
                morphers[0].getImageAtAlpha(alpha, True),
                morphers[1].getImageAtAlpha(alpha, True),
                morphers[2].getImageAtAlpha(alpha, True)
            ])

        cv2.imwrite(f"{output_prefix}_{i}.jpg", outimage)
        print(f"Saved frame {i}/{num_frames}")
        intermediate_images.append(outimage)

    return intermediate_images


# Usage Example:
# morph_images("start.jpg", "end.jpg", num_frames=10, feature_grid_size=8, smoothing=2)
