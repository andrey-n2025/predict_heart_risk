import joblib
import logging
import numpy as np
import pandas as pd
from preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class Model:
    """
    Обёртка над сохранённым полным Pipeline.
    """

    def __init__(self, model_pipeline_path: str):
        try:
            self.pipeline = joblib.load(model_pipeline_path)
            logger.info("Полный Pipeline успешно загружен")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            raise

    def predict_with_info(self, data: pd.DataFrame) -> tuple:
        """
        Возвращает:
        - predictions (np.ndarray)
        - processed_data (pd.DataFrame) — данные после внешней предобработки
        - rows_dropped (int) — сколько строк удалено
        """
        processed_data, rows_dropped = Preprocessor.transform_raw_data(data)
        if processed_data is None or processed_data.empty:
            return np.array([]), processed_data, rows_dropped

        predictions = self.pipeline.predict(processed_data)
        logger.info(f"Предсказания получены для {len(predictions)} объектов")
        return predictions.astype(int), processed_data, rows_dropped