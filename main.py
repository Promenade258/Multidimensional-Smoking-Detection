import cv2
import math
from collections import deque
import mediapipe as mp
from ultralytics import YOLO

# 视频源输入 (0为默认摄像头，可改为视频文件路径)
VIDEO_SOURCE = "1.mp4"

# 模型路径
MODEL_PATH = "./best.pt"

# 置信度
CONFIDENCE = 0.75

# 每隔多少帧检测一次
PROCESS_EVERY_N_FRAMES = 3

# 检测时缩放的宽度
DETECT_WIDTH = 640

# 手靠近嘴部的距离阈值
HAND_MOUTH_DISTANCE = 90

# 香烟靠近嘴部的距离阈值
CIGARETTE_MOUTH_DISTANCE = 90

# 动作分析窗口
ACTION_WINDOW = 20

# 20帧中有8帧手靠近嘴，就认为有吸烟行为
ACTION_THRESHOLD = 8

# ====================================================
# 一、工具函数
# ====================================================

# 计算两点之间的欧氏距离。
def calculate_distance(point1, point2):

    # 获取 point1 的坐标
    if hasattr(point1, 'x'):
        x1 = point1.x
        y1 = point1.y
    else:
        x1 = point1[0]
        y1 = point1[1]

    # 获取 point2 的坐标
    if hasattr(point2, 'x'):
        x2 = point2.x
        y2 = point2.y
    else:
        x2 = point2[0]
        y2 = point2[1]

    dx = x1 - x2
    dy = y1 - y2
    return math.hypot(dx, dy)

# 计算检测框中心点
def get_box_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

# 绘制检测框
def draw_text(frame, text, position, color):
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )

# 缩小图像
def resize_for_detection(frame, target_width):
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame, 1.0, 1.0
    scale = target_width / width
    new_height = int(height * scale)
    resized = cv2.resize(frame, (target_width, new_height))
    scale_x = width / target_width
    scale_y = height / new_height
    return resized, scale_x, scale_y

# 还原图像
def scale_box_to_original(box, scale_x, scale_y):
    x1, y1, x2, y2 = box
    return (int(x1 * scale_x), int(y1 * scale_y),
            int(x2 * scale_x), int(y2 * scale_y))

# ====================================================
# 加载模型
# ====================================================

print("正在加载模型，请稍等...")
# 香烟 + 嘴巴模型
model = YOLO(MODEL_PATH)

# 手指模型
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("模型加载完成")

# ====================================================
# 初始化变量
# ====================================================

action_history = deque(maxlen=ACTION_WINDOW)
frame_count = 0

# 暂停状态
paused = False
current_frame = None

last_method1 = False
last_method2 = False
last_method3 = False
last_method4 = False

last_final_result = "No Smoking"
last_final_color = (0, 255, 0)

last_cigarette_boxes = []
last_mouth_boxes = []
last_mouth_center = None
last_cigarette_center = None
last_left_hand_center = None
last_right_hand_center = None

# ====================================================
# 主循环
# ====================================================

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"视频打开失败！请检查路径：{VIDEO_SOURCE}")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    if not paused:
        success, frame = cap.read()
        if not success:
            print("视频播放结束。")
            break
        frame_count += 1
        current_frame = frame.copy()
    else:
        frame = current_frame.copy()

    height, width = frame.shape[:2]

    # 判断是否执行模型检测
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:

        # 重置本轮结果
        method1_result = False
        method2_result = False
        method3_result = False
        method4_result = False

        cigarette_boxes = []
        mouth_boxes = []
        mouth_center = None
        cigarette_center = None
        hand_near_mouth = False

        left_hand2mouth_distance = None
        right_hand2mouth_distance = None
        left_hand_center = None
        right_hand_center = None

        # 缩小图像
        detect_frame, scale_x, scale_y = resize_for_detection(frame, DETECT_WIDTH)

        # ---- 方法1：检测香烟 ----
        '''
        如果检测到香烟目标 -> True
        '''
        results = model(detect_frame, conf=CONFIDENCE, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                original_box = scale_box_to_original((x1, y1, x2, y2), scale_x, scale_y)

                # 模型类别标签：{0: 'cigarette', 1: 'mouth'}
                if cls == 0:
                    cigarette_boxes.append(original_box)
                    method1_result = True
                elif cls == 1:
                    mouth_boxes.append(original_box)

        # ---- 方法2：检测香烟 + 嘴巴 ----
        '''
        如果同时检测出香烟和嘴巴 -> True
        '''

        if len(mouth_boxes) > 0 and len(cigarette_boxes) > 0:
            method2_result = True

        # 计算嘴巴和香烟中心点
        if mouth_boxes:
            mouth_center = get_box_center(mouth_boxes[0])
        if cigarette_boxes:
            cigarette_center = get_box_center(cigarette_boxes[0])

        # ---- 方法3：手部姿态 + 香烟靠近嘴 ----
        '''
        如果手指中指和食指之间的中心点距离嘴巴低于90px 且 香烟和嘴巴的距离低于50px -> True
        '''
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_result = hands.process(frame_rgb)

        if hands_result.multi_hand_landmarks and hands_result.multi_handedness and mouth_center is not None:
            for hand_landmarks, hand_handedness in zip(hands_result.multi_hand_landmarks,
                                                       hands_result.multi_handedness):
                hand_label = hand_handedness.classification[0].label

                # 获取食指指尖和中指指尖
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # 归一化坐标
                ix = int(index_tip.x * width)
                iy = int(index_tip.y * height)
                mx = int(middle_tip.x * width)
                my = int(middle_tip.y * height)

                # 获取指尖中心点
                finger_center = ((ix + mx) // 2, (iy + my) // 2)
                # tip_distance_px = math.hypot(ix - mx, iy - my)

                # 计算指尖中心点到嘴巴中心点的直线距离
                distance = calculate_distance(finger_center, mouth_center)

                # 计算左手和右手到嘴部的距离
                if hand_label == "Left":
                    left_hand2mouth_distance = distance
                    left_hand_center = finger_center
                    left_index = (ix, iy)
                    left_middle = (mx, my)
                else:
                    right_hand2mouth_distance = distance
                    right_hand_center = finger_center
                    right_index = (ix, iy)
                    right_middle = (mx, my)

                if distance < HAND_MOUTH_DISTANCE:
                    hand_near_mouth = True

            if hand_near_mouth and cigarette_center is not None and mouth_center is not None:
                if calculate_distance(cigarette_center, mouth_center) < CIGARETTE_MOUTH_DISTANCE:
                    method3_result = True

        # ---- 方法4：连续帧动作分析 ----
        '''
        如果连续一段时间内，多次出现“手靠近嘴”的行为->True
        '''
        action_history.append(hand_near_mouth)
        if sum(action_history) >= ACTION_THRESHOLD:
            method4_result = True

        # ---- 综合判断 ----
        '''
        不同方法给不同分数：
        香烟检测：0.5分
        人+香烟：1分
        姿态+香烟靠近嘴：2.5分
        连续动作：1分
        '''
        score = 0
        if method1_result:
            score += 0.5
        if method2_result:
            score += 1
        if method3_result:
            score += 2.5
        if method4_result:
            score += 1

        if score >= 2:
            final_result = "Smoking"
            final_color = (0, 0, 255)
        elif 0.5 < score < 2:
            final_result = "Suspicious Smoking"
            final_color = (0, 165, 255)
        else:
            final_result = "No Smoking"
            final_color = (0, 255, 0)

        # 保存本轮结果
        last_method1 = method1_result
        last_method2 = method2_result
        last_method3 = method3_result
        last_method4 = method4_result
        last_final_result = final_result
        last_final_color = final_color
        last_cigarette_boxes = cigarette_boxes
        last_cigarette_center = cigarette_center
        last_mouth_boxes = mouth_boxes
        last_mouth_center = mouth_center
        last_left_hand_center = left_hand_center
        last_right_hand_center = right_hand_center

    # ----- 绘制最近一次检测的结果 -----

    # 香烟框
    for box in last_cigarette_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        draw_text(frame, "cigarette", (x1, y1 - 5), (0, 0, 255))

    # 嘴巴框
    for box in last_mouth_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        draw_text(frame, "mouth", (x1, y1 - 5), (255, 0, 0))

    # 手部 – 嘴部连线
    if last_mouth_center is not None:

        # 绘制左手连线
        if last_left_hand_center is not None:
            # 画手指指尖圆点
            cv2.circle(frame, left_index, 4, (0, 0, 255), -1)
            cv2.circle(frame, left_middle, 4, (0, 0, 255), -1)
            # 画手指指尖连线
            cv2.line(frame, left_index, left_middle, (255, 255, 255), 2)
            # 画指尖中点圆点
            cv2.circle(frame, last_left_hand_center, 4, (0, 0, 255), -1)
            # 画指尖与嘴巴连线
            cv2.line(frame, last_left_hand_center, last_mouth_center, (255, 255, 255), 2)
            # 计算并显示距离
            dist = calculate_distance(last_left_hand_center, last_mouth_center)
            mid_x = (last_left_hand_center[0] + last_mouth_center[0]) // 2
            mid_y = (last_left_hand_center[1] + last_mouth_center[1]) // 2
            cv2.putText(frame, f"Distance: {int(dist)}px", (mid_x - 40, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 绘制右手连线
        if last_right_hand_center is not None:
            # 画手指指尖圆点
            cv2.circle(frame, right_index, 4, (0, 0, 255), -1)
            cv2.circle(frame, right_middle, 4, (0, 0, 255), -1)
            # 画手指指尖连线
            cv2.line(frame, right_index, right_middle, (255, 255, 255), 2)
            # 画指尖中点圆点
            cv2.circle(frame, last_right_hand_center, 4, (0, 0, 255), -1)
            # 画指尖与嘴巴连线
            cv2.line(frame, last_right_hand_center, last_mouth_center, (255, 255, 255), 2)
            # 计算并显示距离
            dist = calculate_distance(last_right_hand_center, last_mouth_center)
            mid_x = (last_right_hand_center[0] + last_mouth_center[0]) // 2
            mid_y = (last_right_hand_center[1] + last_mouth_center[1]) // 2
            cv2.putText(frame, f"Distance: {int(dist)}px", (mid_x - 40, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 显示四种方法结果和最终判定
    cv2.putText(frame, f"Method1 (cigarette): {last_method1}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Method2 (cig & mouth): {last_method2}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Method3 (hand & cig near mouth): {last_method3}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Method4 (hand near mouth repeatly): {last_method4}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Final: {last_final_result}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_final_color, 2, cv2.LINE_AA)

    # 暂停提示
    if paused:
        cv2.putText(frame, "PAUSED", (frame.shape[1] // 2 - 50, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    cv2.imshow("Smoking Detection System", frame)

    # ESC键退出,空格键暂停
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord(' '):
        paused = not paused

# 释放资源
cap.release()
cv2.destroyAllWindows()
hands.close()
print("程序结束")