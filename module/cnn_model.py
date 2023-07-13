import keras
from keras import layers


def build_model():
    model = keras.Sequential([
        # 특징 추출
        layers.Conv2D(filters=32, kernel_size=(3, 3),
                      activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.Flatten(),
        # 분류기
        layers.Dense(units=32, activation='relu'),
        layers.Dense(units=10, activation='softmax')
    ])
    return model
