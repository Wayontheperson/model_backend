import numpy as np
from PIL import Image


# CIFAR-100 클래스 정보
CIFAR100_CLASSES = [
    # Class names 리스트 (간략화 가능)
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",  # 등등
]


# 모델 초기화
class TFLiteModel:
    def __init__(self, model_path: str):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess_image(self, image_file):
        image = Image.open(image_file.file).convert("RGB")
        image = image.resize((32, 32))  # CIFAR-100 해상도
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=0)  # 배치 차원 추가
        return image

    def predict(self, image_array: np.ndarray):
        self.interpreter.set_tensor(self.input_details[0]["index"], image_array)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]["index"])
        class_index = np.argmax(predictions)
        return {
            "predicted_class": CIFAR100_CLASSES[class_index],
            "confidence": float(predictions[0][class_index]),
        }


# 싱글톤 객체 생성
model = TFLiteModel(model_path="cifar100_model.tflite")
