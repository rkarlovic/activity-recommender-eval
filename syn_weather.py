import random

def get_synthetic_weather():
    weather_conditions = [
        "sunčano i toplo",
        "oblačno i hladno",
        "kišovito i vjetrovito",
        "vedro i ugodno",
        "vruće i sparno",
        "hladno s mogućim snijegom",
        "umjereno s povremenim pljuskovima",
        "sunčano s blagim povjetarcem"
    ]
    return random.choice(weather_conditions)
