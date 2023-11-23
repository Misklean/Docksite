FROM python:3.10

WORKDIR /backend

RUN pip install Flask
RUN pip install flask-cors
RUN pip install gunicorn

COPY . .

ENV FLASK_APP=app.py

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
