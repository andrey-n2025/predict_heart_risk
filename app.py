from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import pandas as pd
import argparse
import os
import io
import json
from pathlib import Path

from model import Model
from preprocessor import Preprocessor
from logging_file import setup_logging

app = FastAPI(
    title="Сервис предсказания риска сердечного приступа",
    description="Графический интерфейс и API для загрузки CSV и получения предсказаний"
)

# Базовая директория проекта
BASE_DIR = Path(__file__).parent.resolve()

# Путь к модели
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "model_pipeline.pkl"
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

setup_logging()
logger = logging.getLogger(__name__)

model = Model(model_pipeline_path=MODEL_PATH)


def process_predictions(data: pd.DataFrame) -> dict:
    """
    Общая функция обработки данных и получения предсказаний.
    Возвращает словарь с результатами для дальнейшего использования в разных эндпоинтах.
    """
    if "id" not in data.columns:
        raise ValueError("В файле отсутствует колонка 'id'")

    logger.info(f"Исходный размер данных: {data.shape}")

    # Сохраняем id в исходном порядке
    original_ids = data["id"].astype(int).values.copy()

    # Получаем предсказания
    y_pred, processed_data, rows_dropped = model.predict_with_info(data)

    if len(y_pred) == 0:
        raise ValueError("После предобработки не осталось валидных записей")

    valid_ids = original_ids[:len(y_pred)]

    result_df = pd.DataFrame({
        "id": valid_ids,
        "prediction": y_pred.astype(int)
    })

    predictions_list = result_df.to_dict(orient="records")

    # CSV в памяти (BytesIO)
    csv_buffer = io.BytesIO()
    result_df.to_csv(csv_buffer, index=False, encoding="utf-8")
    csv_buffer.seek(0)
    csv_bytes = csv_buffer.getvalue()

    return {
        "result_df": result_df,
        "predictions_list": predictions_list,
        "csv_bytes": csv_bytes,
        "total_uploaded": len(data),
        "valid_predictions": len(predictions_list),
        "rows_dropped": rows_dropped
    }


@app.get("/health")
def health():
    return {"status": "OK", "message": "Сервис работает"}


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_web(request: Request, file: UploadFile):
    """
    Веб-интерфейс: отображает результаты + предоставляет скачивание CSV в памяти.
    """
    if not file.filename.endswith(".csv"):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Пожалуйста, загрузите файл в формате CSV"}
        )

    logger.info(f"Получен файл через веб-интерфейс: {file.filename}")

    try:
        content = await file.read()
        data = pd.read_csv(io.StringIO(content.decode("utf-8")))

        result = process_predictions(data)

        context = {
            "request": request,
            "predictions": result["predictions_list"],
            "total_uploaded": result["total_uploaded"],
            "valid_predictions": result["valid_predictions"],
            "rows_dropped": result["rows_dropped"],
            "csv_data": result["csv_bytes"].decode("utf-8"),  # передаём как строку для data-uri
            "filename": file.filename or "predictions.csv"
        }

        return templates.TemplateResponse("index.html", context)

    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {str(e)}", exc_info=True)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Ошибка обработки файла: {str(e)}"}
        )


@app.post("/api/predict")
async def predict_api(file: UploadFile):
    """API: возвращает JSON (streaming)"""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл")

    content = await file.read()
    data = pd.read_csv(io.StringIO(content.decode("utf-8")))

    result = process_predictions(data)

    def generate():
        yield "["
        first = True
        for record in result["predictions_list"]:
            if not first:
                yield ","
            yield json.dumps(record, ensure_ascii=False)
            first = False
        yield "]"

    return StreamingResponse(generate(), media_type="application/json")


@app.post("/api/predict_csv")
async def predict_csv(file: UploadFile):
    """API: возвращает CSV-файл напрямую"""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл")

    content = await file.read()
    data = pd.read_csv(io.StringIO(content.decode("utf-8")))

    result = process_predictions(data)

    return StreamingResponse(
        io.BytesIO(result["csv_bytes"]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename}"}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8999)
    parser.add_argument("--host", type=str, default="localhost")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)