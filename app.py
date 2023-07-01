import pandas as pd
import streamlit as st
from PIL import Image
import pickle


def process_main_page(model):
    show_main_page()
    process_side_bar_inputs(model)


def show_main_page():
    image = Image.open('data/passengers.jpg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Flight satisfaction demo",
        page_icon=image,

    )

    st.write(
        """
        # Предсказание удовлетворенности полётом
        """
    )

    st.image(image)


def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)


def write_prediction(prediction, proba, proba_error):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(proba)

    st.write("## Вероятность ошибки")
    st.write(proba_error)


def load_model():
    model_path = "data/flight_classifier.pickle"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def predict(df_input, model):
    prediction_mapping = {"neutral or dissatisfied": "Клиент не был удовлетворен или поставил нейтральную оценку",
                          "satisfied": "Клиент удовлетворён"}
    
    prediction = prediction_mapping[model.predict(df_input)[0]]
    prediction_text = f"Предсказание: {prediction}"
    prediction_proba = model.predict_proba(df_input)[0]
    round_perc = lambda x: round(x*100, 2)
    proba = round_perc(max(prediction_proba))
    proba_error = round_perc(min(prediction_proba))
    return prediction_text, proba, proba_error

def process_side_bar_inputs(model):
    st.sidebar.header('Заданные пользователем параметры')
    df_input = sidebar_input_features()

    write_user_data(df_input)

    prediction, proba_text, proba_error_text = predict(df_input, model)
    write_prediction(prediction, proba_text, proba_error_text)



def sidebar_input_features():
    gender = st.sidebar.selectbox("Пол", ("Мужской", "Женский"))
    customer_type = st.sidebar.selectbox("Тип клиента", ("Лояльный", "Нелояльный"))
    type_of_travel = st.sidebar.selectbox("Цель поездки", ("Бизнес", "Личная"))
    class_ = st.sidebar.selectbox("Класс", ("Бизнес", "Эко", "Эко-плюс"))

    age = st.sidebar.slider("Возраст", min_value=0, max_value=100, value=20,
                            step=1)



    flight_distance = st.sidebar.slider("Дальность перелёта",
                               min_value=0, max_value=12500, value=3000, step=1)
    dep_delay = st.sidebar.slider("Задержка вылета в минутах",
                               min_value=0, max_value=2880, value=30, step=1)
    arr_delay = st.sidebar.slider("Задержка прилёта в минутах",
                               min_value=0, max_value=2880, value=30, step=1)
    
    infl_wifi = st.sidebar.slider("Оценка интернета на борту",
                               min_value=0, max_value=5, value=3, step=1)
    
    dep_arr_conv = st.sidebar.slider("Оценка удобства времени прилета и вылета",
                               min_value=0, max_value=5, value=3, step=1)
    ease_book = st.sidebar.slider("Оценка удобства онлайн-бронирования",
                               min_value=0, max_value=5, value=3, step=1)
    gate_loc = st.sidebar.slider("Оценка расположения выхода на посадку в аэропорту",
                               min_value=0, max_value=5, value=3, step=1)
    food_drink = st.sidebar.slider("Оценка еды и напитков на борту",
                               min_value=0, max_value=5, value=3, step=1)
    onl_board = st.sidebar.slider("Оценка выбора места в самолете",
                               min_value=0, max_value=5, value=3, step=1)
    seat_comf = st.sidebar.slider("Оценка удобства сиденья",
                               min_value=0, max_value=5, value=3, step=1)
    inflight_entert = st.sidebar.slider("Оценка развлечений на борту",
                               min_value=0, max_value=5, value=3, step=1)
    onboard_serv = st.sidebar.slider("Оценка качества обслуживания на борту",
                               min_value=0, max_value=5, value=3, step=1)
    legroom_serv = st.sidebar.slider("Оценка места в ногах на борту",
                               min_value=0, max_value=5, value=3, step=1)
    baggage_handl = st.sidebar.slider("Оценка обращения с багажом",
                               min_value=0, max_value=5, value=3, step=1)
    checkin_serv = st.sidebar.slider("Оценка регистрации на рейс",
                               min_value=0, max_value=5, value=3, step=1)
    inflight_serv = st.sidebar.slider("Оценка обслуживания на борту",
                               min_value=0, max_value=5, value=3, step=1)
    cleanl = st.sidebar.slider("Оценка чистоты на борту",
                               min_value=0, max_value=5, value=3, step=1)

    gender_mapping = {
        "Мужской": "Male",
        "Женский": "Female"}
    customer_type_mapping = {
        "Лояльный": "Loyal Customer",
        "Нелояльный": "disloyal Customer"}
    class_mapping = {
        "Бизнес": "Business",
        "Эко": "Eco",
        "Эко-плюс": "Eco Plus"}
        
    type_of_travel_maping = {
        "Бизнес": "Business travel",
        "Личная": "Personal travel"}
    

    data = {
        "Gender": gender_mapping[gender],
        "Age": age,
        "Customer Type": customer_type_mapping[customer_type],
        "Type of Travel": type_of_travel_maping[type_of_travel],
        "Class": class_mapping[class_],
        
        "Flight Distance": flight_distance,
        "Departure Delay in Minutes": dep_delay,
        "Arrival Delay in Minutes": arr_delay,
        "Inflight wifi service": infl_wifi,
        "Departure/Arrival time convenient": dep_arr_conv,
        "Ease of Online booking": ease_book,
        "Gate location": gate_loc,
        "Food and drink": food_drink,
        "Online boarding": onl_board,
        "Seat comfort": seat_comf,
        "Inflight entertainment": inflight_entert,
        "On-board service": onboard_serv,
        "Leg room service": legroom_serv,
        "Baggage handling": baggage_handl,
        "Checkin service": checkin_serv,
        "Inflight service": inflight_serv,
        "Cleanliness": cleanl
    }


    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    model = load_model()
    process_main_page(model)
