from fastapi import FastAPI
from app.routes import predict

app = FastAPI()

# 라우터 등록
app.include_router(predict.router)


@app.on_event("startup")
async def startup_event():
    # 워커 시작
    from app.workers.request_worker import start_worker

    start_worker()


@app.get("/")
def read_root():
    return {"message": "TFLite Inference API is running!"}
