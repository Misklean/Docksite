FROM python:3.10

WORKDIR /discord

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "HomeSweetHome.py"]
