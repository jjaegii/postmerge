import numpy as np
import os
from tensorflow import keras
from keras.utils import to_categorical
from PIL import Image
from module.cnn_model import build_model


def run():
    # 이미지와 라벨을 저장할 빈 리스트를 생성합니다.
    X_train = []
    y_train = []

    # 디렉토리를 반복하며 이미지를 로드합니다.
    root_dir = "train"
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            for file_name in os.listdir(dir_path):
                # 이미지 형식에 따라 조정해야 합니다.
                if file_name.endswith(".jpg") or file_name.endswith(".png"):
                    # 이미지를 로드하고 numpy 배열로 변환합니다.
                    image_path = os.path.join(dir_path, file_name)
                    image = Image.open(image_path)
                    image = np.array(image)

                    # 이미지와 라벨을 리스트에 추가합니다.
                    X_train.append(image)
                    # 디렉토리 이름이 라벨이므로, 이를 정수로 변환합니다.
                    y_train.append(int(dir_name))

    # 마지막으로 이미지와 라벨 리스트를 numpy 배열로 변환합니다.
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = []
    y_test = []

    root_dir = "test"
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            for file_name in os.listdir(dir_path):
                # 이미지 형식에 따라 조정해야 합니다.
                if file_name.endswith(".jpg") or file_name.endswith(".png"):
                    # 이미지를 로드하고 numpy 배열로 변환합니다.
                    image_path = os.path.join(dir_path, file_name)
                    image = Image.open(image_path)
                    image = np.array(image)

                    # 이미지와 라벨을 리스트에 추가합니다.
                    X_test.append(image)
                    # 디렉토리 이름이 라벨이므로, 이를 정수로 변환합니다.
                    y_test.append(int(dir_name))

    # 마지막으로 이미지와 라벨 리스트를 numpy 배열로 변환합니다.
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train = X_train/255.
    X_test = X_test/255.

    y_train_o = to_categorical(y_train)
    y_test_o = to_categorical(y_test)

    # 모델 구성
    X_train = X_train.reshape(-1, 28, 28, 1)  # color면 1->3
    X_test = X_test.reshape(-1, 28, 28, 1)

    model = build_model()

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    EPOCHS = 100
    BATCH_SIZE = 256

    model.fit(
        X_train, y_train_o,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test_o),
        verbose=1
    )

    model.save('model/mnist.h5')


if __name__ == "__main__":
    run()
