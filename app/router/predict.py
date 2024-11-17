from fastapi import APIRouter, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from app.models.tflite_model import model

# from app.workers.request_worker import request_queue

router = APIRouter()


@router.post("/predict")
async def predict_endpoint(
    image: UploadFile,
    user_id: str = Form(...),
    background_tasks: BackgroundTasks = None,
):
    async def result_callback(result):
        # 결과를 처리(예: 로그 저장, DB에 기록)
        print(f"Result for user {result['user_id']}: {result}")

    # 요청을 큐에 추가
    await request_queue.put((user_id, image, result_callback))

    # 요청 접수 응답
    return JSONResponse(
        {"message": "Request accepted", "user_id": user_id}, status_code=202
    )
