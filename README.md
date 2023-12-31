# Streamlit Demo

Данный проект демонстрирует работу модели машинного обучения для предсказания удовлетворения пассажиров полётом. Демонстрация проекта происходит с помощью фреймровка [Streamlit](https://www.streamlit.io/). Проект подготовлен в рамках буткемпа "Разработка ML сервиса: от идеи к прототипу", организованного магистратурой "Машинное обучение и высоконагруженные системы" НИУ ВШЭ. 
Используемый датасет содержит информацию о клиентах различных авиакомпаний и содержится в данном репозитории (```flights.csv```)

Интерактивное приложение доступно [по ссылке](https://ml-bootcamp-flights-classifier.streamlit.app/)!

## Структура проекта

- `app.py`: приложение проекта
- `EDA_clients.ipynb`: ноутбук с эксплораторным анализом
- `data`
	- `clients.csv`: используемый датасет
	- `flight_classifier.pickle`: бинарный файл с моделью
- `requirements.txt`: требуемые версии (для поддержки совместимости)

## Для запуска приложения локально через интерфейс командной строки

Выполните следующие команды в корне репозитория проекта

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ streamlit run app.py
```
Приложение будет доступно по адресу http://localhost:8501

