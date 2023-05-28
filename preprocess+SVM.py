
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
import torchvision.datasets.mnist as mnist
plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']

def openImaLabelFile(path): # t10k-labels.idx1-ubyte
    # 指定标签文件路径
    label_file = path

    # 读取标签文件
    with open(label_file, 'rb') as f:
        # 跳过文件头部信息
        f.read(8)
        # 读取标签数量
        num_labels = int.from_bytes(f.read(4), byteorder='big')
        # 读取标签数据
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    # 打印标签数量和前10个标签
    print("标签数量:", num_labels)
    print("前10个标签:", labels[:10])

def openImaFile(path): # t10k-images.idx3-ubyte
    # 指定图像文件路径
    image_file = path

    # 读取图像文件
    with open(image_file, 'rb') as f:
        # 跳过文件头部信息
        f.read(16)
        # 读取图像数量、行数和列数
        num_images = int.from_bytes(f.read(4), byteorder='big')
        num_rows = int.from_bytes(f.read(4), byteorder='big')
        num_cols = int.from_bytes(f.read(4), byteorder='big')
        # 读取图像数据
        images = np.frombuffer(f.read(), dtype=np.uint8).resize(300,300)

    # 可视化前10个图像
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    for i, ax in enumerate(axes.flatten()):
        # 显示图像
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def drawDataRange():
    # 加载 MNIST 数据集
    mnist = fetch_openml('mnist_784')

    # 获取图像数据和标签
    images = mnist.data
    labels = mnist.target

    # 统计每个类别的图像数量
    class_counts = np.bincount(labels.astype(int))

    # 获取类别标签和对应的图像数量
    class_labels = np.arange(10)
    class_images_count = class_counts[class_labels]

    # 可视化每个类别的图像数量
    plt.figure(figsize=(8, 4))
    plt.bar(class_labels, class_images_count)
    plt.xlabel('类别')
    plt.ylabel('图像数量')
    plt.title('MNIST 数据集类别图像数量')
    plt.xticks(class_labels)
    plt.show()

def mnistSVM(kernelCategory):
    print(kernelCategory)
    # 1. 读取 MNIST 数据
    # 使用 fetch_openml 函数从 OpenML 上获取 MNIST 数据集，包括图像数据和标签
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    # 获取图像数据和标签
    X = mnist.data
    y = mnist.target

    # 2. 划分训练集和测试集
    # 使用 train_test_split 函数将数据划分为训练集和测试集，比例为 7:3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. 特征缩放
    # 使用 MinMaxScaler 对图像数据进行归一化处理，将像素值缩放到 [0, 1] 范围
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 数据降维
    pca = PCA(n_components=0.95)  # 保留 95% 的方差
    X_train_scaled = pca.fit_transform(X_train_scaled)
    X_test_scaled = pca.transform(X_test_scaled)

    # 5. 构建支持向量机模型
    # 使用 SVC 创建支持向量机模型，选择适当的核函数（这里选择径向基核）
    svm_model = SVC(kernel=kernelCategory)
    svm_model.fit(X_train_scaled, y_train)

    # 6. 模型评估
    # 在测试集上进行预测
    y_pred = svm_model.predict(X_test_scaled)

    # 计算模型的准确率、召回率、F1 值等指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("模型评估结果：")
    print("准确率：", accuracy)
    print("召回率：", recall)
    print("F1 值：", f1)



train_set = (
    mnist.read_image_file('./train-images-idx3-ubyte/train-images.idx3-ubyte'),
    mnist.read_label_file('./train-labels-idx1-ubyte/train-labels.idx1-ubyte')
        )
test_set = (
    mnist.read_image_file('./t10k-images-idx3-ubyte/t10k-images.idx3-ubyte'),
    mnist.read_label_file('./t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')
        )
print("training set :",train_set[0].size())
print("test set :",test_set[0].size())

def convert_to_img(train=True):
    if(train):
        f=open('train.txt','w')
        data_path='./train/'
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(train_set[0],train_set[1])):
            img_path=data_path+str(i)+'.jpg'
            io.imsave(img_path,img.numpy())
            f.write(img_path+' '+str(label)+'\n')
        f.close()
    else:
        f = open('test.txt', 'w')
        data_path = './test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(test_set[0],test_set[1])):
            img_path = data_path+ str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label) + '\n')
        f.close()

# convert_to_img(True)#转换训练集
# convert_to_img(False)#转换测试集
# mnistSVM('rbf') # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
mnistSVM('linear')
mnistSVM('poly')
mnistSVM('sigmoid')
mnistSVM('precomputed')

# # drawDataRange()
# openImaFile('./train-images-idx3-ubyte/train-images.idx3-ubyte')
# openImaLabelFile('./train-labels-idx1-ubyte/train-labels.idx1-ubyte')
# openImaFile('./t10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
# openImaLabelFile('./t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')
