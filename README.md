
---

# BINANCE SMART-EMA SINGLE TOKEN TRADER

A fast, fully automated, GUI-based single-token EMA strategy optimizer and trader for Binance Spot.

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
* [6. Notes](#notes)

---

## <a name="features"></a>1. Features

* One-click install (all dependencies in requirements.txt)
* GUI (PySide6) for entering keys, symbol, backtest settings, etc.
* Randomizes and finds optimal EMAs and take-profit for best win rate
* Runs continuous auto re-optimization and live trading
* Fully adjustable (pair, lookback, TP range, random test count)
* Designed to be executable-ready (.exe builds supported)

---

## <a name="prerequisites"></a>2. Prerequisites

* Python 3.9 or higher (recommended: Python 3.10/3.11)
* [Binance API keys](https://www.binance.com/en/my/settings/api-management)
* Internet access

---

## <a name="setup-instructions"></a>3. Setup Instructions

### <a name="chromeos"></a>A. Linux on ChromeOS (Crostini)

1. **Open the Terminal app** (`Ctrl + Alt + T` or via Launcher).
2. **Update and install Python:**

   ```sh
   sudo apt update
   sudo apt install python3 python3-pip python3-venv git -y
   ```
3. **(Recommended) Create a Python virtual environment:**

   ```sh
   python3 -m venv smartemaenv
   source smartemaenv/bin/activate
   ```
4. **Clone the repository:**

   ```sh
   git clone https://github.com/<yourusername>/BINANCE_SMART-EMA_SINGLE_TOKEN_TRADER.git
   cd BINANCE_SMART-EMA_SINGLE_TOKEN_TRADER
   ```
5. **Install Python dependencies:**

   ```sh
   pip install --upgrade pip
   pip install -r requirements.txt --break-system-packages
   ```

   > If you get errors about system packages on ChromeOS, use `--break-system-packages` as above.

---

### <a name="ubuntu"></a>B. Ubuntu Linux

1. **Open Terminal** (`Ctrl + Alt + T`).
2. **Update and install Python:**

   ```sh
   sudo apt update
   sudo apt install python3 python3-pip python3-venv git -y
   ```
3. **(Recommended) Create a Python virtual environment:**

   ```sh
   python3 -m venv smartemaenv
   source smartemaenv/bin/activate
   ```
4. **Clone the repository:**

   ```sh
   git clone https://github.com/<yourusername>/BINANCE_SMART-EMA_SINGLE_TOKEN_TRADER.git
   cd BINANCE_SMART-EMA_SINGLE_TOKEN_TRADER
   ```
5. **Install dependencies:**

   ```sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

### <a name="windows11"></a>C. Windows 11

1. **Install [Python 3.11+](https://www.python.org/downloads/windows/):**

   * Download and install, check "Add Python to PATH" during setup.
2. **Install [Git for Windows](https://git-scm.com/download/win):**

   * Use default options.
3. **Open "Command Prompt" or "Windows Terminal":**

   * Hit `Win + R`, type `cmd`, press Enter.
4. **(Recommended) Create a virtual environment:**

   ```cmd
   python -m venv smartemaenv
   smartemaenv\Scripts\activate
   ```
5. **Clone the repo:**

   ```cmd
   git clone https://github.com/<yourusername>/BINANCE_SMART-EMA_SINGLE_TOKEN_TRADER.git
   cd BINANCE_SMART-EMA_SINGLE_TOKEN_TRADER
   ```
6. **Install dependencies:**

   ```cmd
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## <a name="running"></a>4. Running the Bot

**From inside the repo folder and (optionally) your virtual environment:**

```sh
python BINANCE_SMART-EMA_SINGLE_TOKEN_TRADER.py
```

* The GUI should open. Enter your Binance API key/secret, trading symbol (default BNBUSDT), and other settings.
* The config is saved after clicking "Save".
* "Start Live Trading" begins auto-optimization and live bot execution.

---

## <a name="vscode"></a>5. Setting Up Visual Studio Code (All Platforms)

**Install [Visual Studio Code](https://code.visualstudio.com/):**

* Download and install for your platform.

**Open your project:**

* Click "File" → "Open Folder" and select your repo folder.

**Install the Python extension:**

* Click the Extensions icon (four squares, left sidebar).
* Search for "Python" and install the Microsoft extension.

**(Optional but recommended)**

* Open a terminal inside VS Code: `Ctrl + `` (backtick)
* Activate your virtual environment:

  * **Linux/macOS:** `source smartemaenv/bin/activate`
  * **Windows:** `smartemaenv\Scripts\activate`
* Run the bot with:

  ```sh
  python BINANCE_SMART-EMA_SINGLE_TOKEN_TRADER.py
  ```

---

## <a name="notes"></a>6. Notes & Troubleshooting

* **API keys:** Keep your API keys secret! Store with "Read" and "Trade" permissions only (never enable withdrawal).
* **First run:** On first run, a `config.json` is created/saved in the same folder for your settings.
* **Linux dependencies:** If you get errors for Qt or missing display, install:

  ```sh
  sudo apt install libxcb-cursor0 libxcb-xinerama0
  ```
* **Upgrading:** Pull latest code with `git pull` inside your repo folder.
* **EXE:** If building an EXE for Windows, use [PyInstaller](https://pyinstaller.org/en/stable/) (ask for special instructions if needed).

---

## Enjoy safe trading!


Thanks for taking a look at this code and I hope it works just as well for you as it has for me!

If you would like to donate to keep this project afloat and maybe even fund late night MiGoreng binges whilst coding enhancements, you can do so using this BNB Wallet QR code. Any amount is appreciated but not mandatory!

<img width="225" height="317" alt="image" src="https://github.com/user-attachments/assets/98530f17-47ee-4f3c-be45-3dc9f2bda613" />


Do you need a referral link to make the most of Binance giving away USD?

https://www.binance.com/activity/referral-entry/CPA?ref=CPA_00V9WDVAJY

Referral Code: CPA_00V9WDVAJY

Best of luck with your Crypto journey!! I hope it helps :)

(Don't forget to check this repository for regular updates)

Cheers
If you find this project useful, please star ⭐ the repo.

---

