import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Beispieldaten laden (CSV-Datei muss im Arbeitsverzeichnis liegen)
data = pd.read_csv('HistoricalData_Nasdaq.csv')

# Technische Indikatoren berechnen (z.B. RSI, SMA, MACD)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

# Berechne RSI und SMA
data['RSI'] = calculate_rsi(data)
data['SMA_20'] = calculate_sma(data, 20)
data['SMA_50'] = calculate_sma(data, 50)

# Fülle fehlende Werte
data.fillna(0, inplace=True)

# Features und Labels definieren
features = ['RSI', 'SMA_20', 'SMA_50']
X = data[features]
y = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)  # 1 = Preis steigt, 0 = Preis fällt

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Neuronales Netz erstellen
model = Sequential()
model.add(Dense(64, input_dim=len(features), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Kompilieren des Modells
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modell trainieren
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# TradingAgent-Klasse
class TradingAgent:
    def __init__(self, model, stop_loss_pct=0.02, take_profit_pct=0.02):
        self.model = model
        self.position = None
        self.stop_loss = None
        self.take_profit = None
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.profit = 0

    def predict(self, features):
        prediction = self.model.predict(features)
        return prediction[0][0]  # Vorhersage als einzelner Wert

    def open_position(self, price):
        self.position = 'buy'
        self.stop_loss = price * (1 - self.stop_loss_pct)
        self.take_profit = price * (1 + self.take_profit_pct)
        print(f"Position eröffnet bei {price}. Stop-Loss: {self.stop_loss}, Take-Profit: {self.take_profit}")

    def close_position(self, price):
        if price >= self.take_profit:
            self.profit += price - self.take_profit
            print(f"Position geschlossen mit Take-Profit bei {price}")
        elif price <= self.stop_loss:
            self.profit += self.stop_loss - price
            print(f"Position geschlossen mit Stop-Loss bei {price}")
        self.position = None

# Simulationscode für Trading
agent = TradingAgent(model)

for i in range(len(X_test)):
    features = X_test.iloc[i].values.reshape(1, -1)
    price = data['Close'].iloc[i + len(X_train)]

    if agent.position is None:  # Keine offene Position
        prediction = agent.predict(features)
        if prediction > 0.5:  # Kaufsignal
            agent.open_position(price)

    elif agent.position == 'buy':  # Position halten
        if price >= agent.take_profit or price <= agent.stop_loss:
            agent.close_position(price)

# Belohnungsfunktion
def reward_function(profit):
    # Positive Belohnung bei Profit, negative bei Verlust
    return profit if profit > 0 else -1

for i in range(len(X_test)):
    features = X_test.iloc[i].values.reshape(1, -1)
    price = data['Close'].iloc[i + len(X_train)]

    if agent.position is None:  # Öffne Position
        prediction = agent.predict(features)
        if prediction > 0.5:
            agent.open_position(price)

    elif agent.position == 'buy':  # Halte Position
        if price >= agent.take_profit or price <= agent.stop_loss:
            agent.close_position(price)
            reward = reward_function(agent.profit)
            # Hier könnte eine Anpassung oder Lernschleife eingefügt werden

# Backtesting-Funktion, um die Performance der Strategie zu bewerten
def backtest(agent, data):
    trades = 0
    total_profit = 0
    for i in range(len(data) - 1):
        price = data['Close'].iloc[i]
        if agent.position is None:
            agent.open_position(price)
        else:
            if price >= agent.take_profit or price <= agent.stop_loss:
                agent.close_position(price)
                total_profit += agent.profit
                trades += 1
    return total_profit, trades

# Backtest durchführen
profit, trades = backtest(agent, data)
print(f"Total Profit: {profit}, Trades: {trades}")
