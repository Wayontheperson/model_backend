import asyncio
from app.models.tflite_model import model

# 비동기 요청 큐
request_queue = asyncio.Queue()


async def process_requests():
    while True:
        user_id, image_file, result_callback = await request_queue.get()
        try:
            # 이미지 처리 및 추론
            image_array = model.preprocess_image(image_file)
            result = model.predict(image_array)
            result["user_id"] = user_id
            # 콜백 호출
            await result_callback(result)
        except Exception as e:
            await result_callback({"error": str(e), "user_id": user_id})
        finally:
            request_queue.task_done()


def start_worker():
    # 워커 태스크 시작
    asyncio.create_task(process_requests())
