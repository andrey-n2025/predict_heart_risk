import pandas as pd
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Класс для внешней предобработки сырых данных перед подачей в pipeline.
    """
    COLS_FOR_DROP = ['Unnamed: 0', 'CK-MB', 'Troponin', 'id']

    @classmethod
    def transform_raw_data(cls, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], int]:
        """
        Выполняет внешние преобразования и возвращает:
        - обработанный DataFrame (готовый для pipeline.predict())
        - количество удалённых строк из-за пропусков
        """
        try:
            if data.empty:
                logger.warning("Входные данные пустые")
                return None, 0

            df = data.copy()
            initial_rows = df.shape[0]

            # Удаление технических колонок
            cols_to_drop = [col for col in cls.COLS_FOR_DROP if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                logger.info(f"Удалены колонки: {cols_to_drop}")

            # Удаление строк с пропусками
            df = df.dropna()
            rows_dropped = initial_rows - df.shape[0]
            if rows_dropped > 0:
                logger.info(f"Удалено {rows_dropped} строк с пропущенными значениями")

            logger.info(f"Предобработка завершена. Shape: {df.shape}")
            return df, rows_dropped

        except Exception as e:
            logger.error(f"Ошибка предобработки: {str(e)}", exc_info=True)
            raise