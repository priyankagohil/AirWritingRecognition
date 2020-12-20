import cv2
import numpy as np
import copy
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

rect1_tl = (320, 140)
rect2_tl = (320, 240)
rect3_tl = (320, 340)
rect4_tl = (240, 270)
rect5_tl = (400, 270)

height = 30
width = 30

"""CNN architecture of the model"""


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 26)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Adding sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def get_histogram(frame):
    roi1 = frame[rect1_tl[1]:rect1_tl[1] + width, rect1_tl[0]:rect1_tl[0] + height]
    roi2 = frame[rect2_tl[1]:rect2_tl[1] + width, rect2_tl[0]:rect2_tl[0] + height]
    roi3 = frame[rect3_tl[1]:rect3_tl[1] + width, rect3_tl[0]:rect3_tl[0] + height]
    roi4 = frame[rect4_tl[1]:rect4_tl[1] + width, rect4_tl[0]:rect4_tl[0] + height]
    roi5 = frame[rect5_tl[1]:rect5_tl[1] + width, rect5_tl[0]:rect5_tl[0] + height]
    roi = np.concatenate((roi1, roi2, roi3, roi4, roi5), axis=0)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    return cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])


def draw_rectangles(frame=0):
    frame_with_rect = frame
    cv2.rectangle(frame_with_rect, rect1_tl, tuple(np.array(rect1_tl) + np.array((height, width))), (0, 0, 255), 1)
    cv2.rectangle(frame_with_rect, rect2_tl, tuple(np.array(rect2_tl) + np.array((height, width))), (0, 0, 255), 1)
    cv2.rectangle(frame_with_rect, rect3_tl, tuple(np.array(rect3_tl) + np.array((height, width))), (0, 0, 255), 1)
    cv2.rectangle(frame_with_rect, rect4_tl, tuple(np.array(rect4_tl) + np.array((height, width))), (0, 0, 255), 1)
    cv2.rectangle(frame_with_rect, rect5_tl, tuple(np.array(rect5_tl) + np.array((height, width))), (0, 0, 255), 1)
    return frame_with_rect


def get_mask(frame, histogram):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([frame_hsv], [0, 1], histogram, [0, 180, 0, 256], 1)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask = cv2.filter2D(mask, -1, kernel)

    kernel1 = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.bilateralFilter(mask, 5, 75, 75)

    return mask


def get_max_contour(mask):
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    max = 0
    mi = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 1500:
            max = area
            mi = i
    return contours[mi]


def draw_defects(frame_with_rect, max_contour, hull):
    defects = cv2.convexityDefects(max_contour, hull)

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])
        cv2.line(frame_with_rect, start, far, [255, 0, 0], 2)
        cv2.line(frame_with_rect, far, end, [0, 255, 0], 2)
        cv2.circle(frame_with_rect, far, 5, [0, 0, 255], -1)


def get_centroid(contour):
    m = cv2.moments(contour)
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    return cx, cy


def get_farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None


def get_ROI(canvas):
    gray = cv2.bitwise_not(canvas)
    ret, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
    _, ctrs, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for i in range(len(ctrs)):
        x, y, w, h = cv2.boundingRect(ctrs[i])
        areas.append((w * h, i))

    def sort_second(val):
        return val[0]

    areas.sort(key=sort_second, reverse=True)
    x, y, w, h = cv2.boundingRect(ctrs[areas[1][1]])
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 0), 1)
    roi = gray[y:y + h, x:x + w]
    return roi


def character_prediction(roi, model):
    """Predicts character written with image processing"""
    img = cv2.resize(roi, (28, 28))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = Image.fromarray(img)

    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        normalize
    ])

    p_img = preprocess(img)

    model.eval()
    p_img = p_img.reshape([1, 1, 28, 28]).float()
    output = model(torch.transpose(p_img, 2, 3))
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds


def main():
    video = cv2.VideoCapture(0)
    canvas = np.zeros((720, 1280), np.uint8)
    far_points = []

    pressed = False
    is_drawing = False
    made_prediction = False
    # Creating the model
    model = Cnn()

    model.load_state_dict(torch.load('model_emnist.pt', map_location='cpu'))

    # Actions to perform with each key
    while True:
        _, frame = video.read()
        frame = cv2.flip(frame, flipCode=1)
        original_frame = copy.deepcopy(frame)
        original_frame = draw_rectangles(original_frame)
        canvas[:, :] = 255

        key = cv2.waitKey(1)

        # ready to draw
        if key & 0xFF == ord('s'):
            pressed = True
            histogram = get_histogram(frame)

        # To start drawing
        if key & 0xFF == ord('d'):
            is_drawing = True

        # To clear drawing
        if key & 0xFF == ord('c'):
            canvas[:, :] = 255
            is_drawing = False
            far_points.clear()
            made_prediction = False

        if is_drawing:
            if len(far_points) > 100:
                far_points.pop(0)
            far_points.append(far)
            for i in range(len(far_points) - 1):
                cv2.line(original_frame, far_points[i], far_points[i + 1], (255, 5, 255), 20)
                cv2.line(canvas, far_points[i], far_points[i + 1], (0, 0, 0), 20)

        # To predict the character drawn
        if key & 0xFF == ord('p'):
            is_drawing = False
            roi = get_ROI(canvas)
            prediction = character_prediction(roi, model)
            print(prediction)
            made_prediction = True
            name = str(prediction) + '.jpg'
            cv2.imwrite(name, roi)

        if pressed:
            mask = get_mask(frame, histogram)
            max_contour = get_max_contour(mask)
            hull = cv2.convexHull(max_contour, returnPoints=False)
            draw_defects(original_frame, max_contour, hull)
            defects = cv2.convexityDefects(max_contour, hull)
            far = get_farthest_point(defects, max_contour, get_centroid(max_contour))
            cv2.circle(original_frame, far, 10, [0, 200, 255], -1)

        if made_prediction:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(original_frame, 'Character written : ' + chr(prediction + 65), (10, 500), font, 4,
                        (255, 255, 255), 2, cv2.LINE_AA)

        # To quit the drawing
        if key & 0xFF == ord('q'):
            break
        cv2.imshow('frame', original_frame)

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
