## Команда для запуска приложения
**uvicorn app:app --reload --port 8000**

Эндпоинт для отправки запроса и получения ответа:
HTTP:POST http://127.0.0.1:8000/ask

Payload: {
  question: "put your question here..."
}
