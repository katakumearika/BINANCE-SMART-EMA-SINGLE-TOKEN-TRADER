Here’s an **updated README** that matches the **latest version of your bot**, including:

* ✅ EMA logging every minute
* 🕒 Signal confirmation only on 15-minute candle close
* 🧠 Automatic re-optimization after each SELL
* 🖥️ GUI with Initial & Re-opt trials
* 💡 Optimized live trading using best EMA parameters

---

# BINANCE SMART-EMA SINGLE TOKEN TRADER

A fast, fully automated, GUI-based **EMA strategy optimizer and live trading bot** for Binance Spot.

<img width="744" height="556" alt="image" src="https://github.com/user-attachments/assets/b2964c80-f1b9-411d-9185-a51c17a1d28b" />

---

## Table of Contents

* [1. Features](#features)
* [2. Prerequisites](#prerequisites)
* [3. Setup Instructions](#setup-instructions)

  * [A. Linux on ChromeOS](#chromeos)
  * [B. Ubuntu Linux](#ubuntu)
  * [C. Windows 11](#windows11)
* [4. Running the Bot](#running)
* [5. Visual Studio Code Setup](#vscode)
* [6. Strategy Behavior](#strategy)
* [7. Notes](#notes)

---

## <a name="features"></a>1. Features

* ✅ GUI built with PySide6 — no CLI needed.
* 🧮 **Optimizes Fast/Mid/Slow EMAs and TP** automatically for best win rate.
* 🔁 **Auto re-optimization after each SELL** — no restart needed.
* ⏳ **Live EMA trend logging every 1 minute** (to monitor trend evolution).
* 🕒 **Crossover detection only on 15-minute candle close** — avoids fake signals.
* 💰 Fully adjustable (pair, lookback period, TP range, initial & reopt trials).
* 🧰 Executable-ready (PyInstaller supported for Windows `.exe` builds).

---

## <a name="prerequisites"></a>2. Prerequisites

* Python 3.9 or higher (3.10/3.11 recommended)
* [Binance API keys](https://www.binance.com/en/my/settings/api-management) with **Read** and **Trade** permissions
* Internet access

---

## <a name="setup-instructions"></a>3. Setup Instructions

### <a name="chromeos"></a>A. Linux on ChromeOS (Crostini)

```sh
sudo apt update
sudo apt install python3 python3-pip python3-venv git -y
python3 -m venv smartemaenv
source smartemaenv/bin/activate
git clone https://github.com/<yourusername>/BINANCE_SMART-EMA_SINGLE_TOKEN_TRADER.git
cd BINANCE_SMART-EMA_SINGLE_TOKEN_TRADER
pip install --upgrade pip
pip install -r requirements.txt --break-system-packages
```

---

### <a name="ubuntu"></a>B. Ubuntu Linux

```sh
sudo apt update
sudo apt install python3 python3-pip python3-venv git -y
python3 -m venv smartemaenv
source smartemaenv/bin/activate
git clone https://github.com/<yourusername>/BINANCE_SMART-EMA_SINGLE_TOKEN_TRADER.git
cd BINANCE_SMART-EMA_SINGLE_TOKEN_TRADER
pip install --upgrade pip
pip install -r requirements.txt
```

---

### <a name="windows11"></a>C. Windows 11

```cmd
python -m venv smartemaenv
smartemaenv\Scripts\activate
git clone https://github.com/<yourusername>/BINANCE_SMART-EMA_SINGLE_TOKEN_TRADER.git
cd BINANCE_SMART-EMA_SINGLE_TOKEN_TRADER
pip install --upgrade pip
pip install -r requirements.txt
```

---

## <a name="running"></a>4. Running the Bot

Run from inside the repo folder:

```sh
python triple_ema_gui_bot.py
```

* A GUI window will open.
* Enter your **Binance API key**, trading symbol (e.g., `BNBUSDT`), lookback, and trial counts.
* Click **💾 Save Config** to store settings.
* Press **▶️ Start** to begin optimization and live trading.

The bot will:

* 🔥 Run initial optimization to find best EMA & TP.
* 📈 Log EMA trend every minute.
* 📊 Wait for a 15-minute candle close to confirm crossover.
* 💵 Place BUY/SELL orders using Binance Spot API.
* 🔁 Re-optimize after each SELL using the “Re-Opt Trials” count.

---

## <a name="vscode"></a>5. Visual Studio Code Setup (All Platforms)

```sh
# Optional but recommended
code .
source smartemaenv/bin/activate  # or smartemaenv\Scripts\activate on Windows
python triple_ema_gui_bot.py
```

Install the **Python extension** in VS Code for best experience.

---

## <a name="strategy"></a>6. Strategy Behavior

### 📊 1-Minute EMA Logging

The bot updates and logs **Fast, Mid, and Slow EMA** every 60 seconds using Binance ticker data.

### 🕒 15-Minute Candle Confirmation

The bot only confirms crossovers on **15-minute candle close**, avoiding fake or premature signals.

### 🔁 Auto Re-Optimization

After each SELL (TP or crossdown), the bot:

* Fetches fresh OHLCV data,
* Runs the optimizer using `Re-Opt Trials` count,
* Updates live trading parameters without restarting.

---

## <a name="notes"></a>7. Notes & Troubleshooting

* ⚠️ **API Keys:** Never share your keys. Enable only “Spot Trading” permission.
* 🖥️ First run creates a local `ema_bot_config.json` to save your settings.
* 🧰 Missing Qt libraries on Linux?

  ```sh
  sudo apt install libxcb-cursor0 libxcb-xinerama0
  ```
* 🔄 Update with:

  ```sh
  git pull
  ```
* 🪟 EXE builds supported via [PyInstaller](https://pyinstaller.org/en/stable/).

---

## 💸 Sponsor / Donate

If you like this project, you can support development in two ways:

1. **Register with my Binance referral link:**
   👉 [Binance Referral — CPA_00V9WDVAJY](https://www.binance.com/activity/referral-entry/CPA?ref=CPA_00V9WDVAJY)

2. **Donate USDT or any ERC20/BEP20 token:**
   `0xc22f994de2a5b55b359221b51a813b999c713751`

<img src="https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=0xc22f994de2a5b55b359221b51a813b999c713751" alt="Donate USDT QR Code" width="200"/>

⭐ If you find this project useful, please star the repo!

---

✅ *Stable, fast, optimized, fully automated. EMA-based. One token. Real edge.*
