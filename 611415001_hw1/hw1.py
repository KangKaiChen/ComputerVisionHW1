import numpy as np
import cv2

# 讀取RGB圖像

ques = input("Enter a number: 1 is car and 2 is liberty ")
if ques == "2": 
    img = cv2.imread('liberty.png' )
if ques == "1": 
    img = cv2.imread('car.png' )


# 將圖像轉換為灰階圖像
def get_red(img):
    redImg = img[:,:,2]
    return redImg

def get_green(img):
    greenImg = img[:,:,1]
    return greenImg

def get_blue(img):
    blueImg = img[:,:,0]
    return blueImg


# 取得RGB圖像中的三個顏色通道
blue = get_blue(img)
green = get_green(img)
red = get_red(img)

# 將三個通道加權平均，得到灰階圖像
gray_img = 0.2989 * red + 0.5870 * green + 0.1140 * blue
gray_img = gray_img.astype(np.uint8)

# 定義邊緣檢測核
edge_detector = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# 在影像周圍填充0
img_padded = np.pad(gray_img, ((1,1),(1,1)), 'constant', constant_values=(0,0))
# 創建一個空的輸出影像
conv_img = np.zeros_like(gray_img)

# 迭代計算卷積
for i in range(1, img_padded.shape[0]-1):
    for j in range(1, img_padded.shape[1]-1):
        # 取出3x3區域
        region = img_padded[i-1:i+2, j-1:j+2]
        # 執行卷積
        conv_val = (region * edge_detector).sum()
        conv_img[i-1, j-1] = max(0, conv_val)
        
        
pool_img = np.zeros((conv_img.shape[0]//2, conv_img.shape[1]//2), dtype=np.uint8)      
        
# 執行 Max Pooling 運算
for i in range(0, conv_img.shape[0]-1, 2):
    for j in range(0, conv_img.shape[1]-1, 2):
        pool_val = np.max(conv_img[i:i+2, j:j+2])  #求取 2x2 的區域中的最大值，並將其儲存到 pool_val
        pool_img[i//2, j//2] = pool_val       
# 將輸出影像限制在0到255之間

# 手動二值化
threshold = 128
binary_img = np.zeros(pool_img.shape, dtype=np.uint8)
binary_img[pool_img >= threshold] = 255

# 顯示結果
# 顯示灰階圖像
cv2.imshow('Gray Image', gray_img)
cv2.imshow('Edge Detection', conv_img)
cv2.imshow('Max Pooling', pool_img)
cv2.imshow('Binary Image', binary_img)


if ques == str(1): 
    cv2.imwrite("car_Q1.png",gray_img )
    cv2.imwrite("car_Q2.png",conv_img )
    cv2.imwrite("car_Q3.png",pool_img )
    cv2.imwrite("car_Q4.png",binary_img )
    
else: 
    cv2.imwrite("liberty_Q1.png",gray_img )
    cv2.imwrite("liberty_Q2.png",conv_img )
    cv2.imwrite("liberty_Q3.png",pool_img )
    cv2.imwrite("liberty_Q4.png",binary_img ) 
cv2.waitKey(0)
cv2.destroyAllWindows()
