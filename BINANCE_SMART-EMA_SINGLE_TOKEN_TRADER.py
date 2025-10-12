import sys
import os
import json
import asyncio
import time
import random
import pandas as pd
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext
import concurrent.futures
import websockets
from binance.spot import Spot

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QProgressBar, QFileDialog, QPlainTextEdit
)
from PySide6.QtCore import Qt, QThread, Signal

getcontext().prec = 16
CONFIG_FILE = "ema_bot_config.json"
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# ------------------- Trading Thread ------------------- #
class TradingThread(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int)
    finished_signal = Signal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._running = True
        self.fast = None
        self.mid = None
        self.slow = None
        self.tp_val = None
        self.client = None
        self.symbol = None
        self.interval = None
        self.lookback = None
        self.initial_trials = None
        self.reopt_trials = None
        self.min_qty = None
        self.step_size = None
        self.min_notional = None
        self.ema_fast = None
        self.ema_mid = None
        self.ema_slow = None

    def stop(self):
        self._running = False

    def log(self, msg):
        self.log_signal.emit(msg)

    def run(self):
        asyncio.run(self.main_logic())

    # -------- Helpers -------- #
    def get_balance(self, asset):
        acc = self.client.account()
        for b in acc['balances']:
            if b['asset'] == asset:
                return Decimal(str(b['free']))
        return Decimal('0')

    async def async_balance(self, asset):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, lambda: self.get_balance(asset))

    def trend_str(self, curr, prev):
        if curr > prev: return "UP"
        if curr < prev: return "DOWN"
        return "FLAT"

    def get_symbol_info(self):
        info = self.client.exchange_info(symbol=self.symbol)
        filters = {f['filterType']: f for f in info['symbols'][0]['filters']}
        self.min_qty = Decimal(filters['LOT_SIZE']['minQty'])
        self.step_size = Decimal(filters['LOT_SIZE']['stepSize'])
        self.min_notional = Decimal(filters.get('MIN_NOTIONAL', {}).get('minNotional', '0'))

    def quantize_qty(self, qty):
        dqty = Decimal(str(qty))
        dstep = Decimal(str(self.step_size))
        return float((dqty // dstep) * dstep)

    def calc_buy_qty(self, usdt, price):
        d_price = Decimal(str(price))
        d_usdt = Decimal(str(usdt))
        qty = (d_usdt * (Decimal('1') - Decimal('0.001'))) / d_price
        qty = self.quantize_qty(qty)
        if Decimal(str(qty)) < self.min_qty or Decimal(str(qty)) * d_price < self.min_notional:
            return None
        return qty

    def calc_sell_qty(self, asset, price):
        d_asset = Decimal(str(asset))
        d_price = Decimal(str(price))
        qty = self.quantize_qty(d_asset)
        if Decimal(str(qty)) < self.min_qty or Decimal(str(qty)) * d_price < self.min_notional:
            return None
        return qty

    def fetch_ohlcv(self, hours, extra_candles=150):
        end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ts = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp() * 1000)
        klines, current = [], start_ts
        loops = 0
        while current < end_ts and loops < 200:
            batch = self.client.klines(symbol=self.symbol, interval=self.interval,
                                  startTime=current, endTime=end_ts, limit=1000)
            if not batch:
                break
            klines.extend(batch)
            next_time = batch[-1][0] + 1
            if next_time <= current:
                break
            current = next_time
            time.sleep(0.15)
            loops += 1
            if len(klines) >= int((hours * 60 // 15) + extra_candles):
                break
        df = pd.DataFrame(klines, columns=[
            'open_time','open','high','low','close','volume',
            'close_time','quote_vol','n_trades','taker_base_vol','taker_quote_vol','ignore'
        ])
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df.set_index('open_time', inplace=True)
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        return df

    def triple_ema_backtest(self, df, fast, mid, slow, tp_value):
        data = df.copy()
        data['fast_ema'] = data['ohlc4'].ewm(span=fast, adjust=False).mean()
        data['mid_ema']  = data['ohlc4'].ewm(span=mid, adjust=False).mean()
        data['slow_ema'] = data['ohlc4'].ewm(span=slow, adjust=False).mean()
        data[['prev_fast','prev_mid','prev_slow']] = data[['fast_ema','mid_ema','slow_ema']].shift(1)

        balance, asset = Decimal('1000'), Decimal('0')
        entry_price = None
        trades, wins = 0, 0
        FEE = Decimal('0.001')

        for _, r in data.iterrows():
            crossup = (r.prev_fast < r.prev_mid and r.fast_ema >= r.mid_ema)
            slow_trending_up = r.slow_ema >= r.prev_slow  # 🆕 new filter
            cond_above = (r.mid_ema > r.slow_ema) or (r['open'] > r.slow_ema) or (r['close'] > r.slow_ema)
            crossdown = ((r.prev_fast > r.prev_mid and r.fast_ema <= r.mid_ema) or
                        (r.prev_fast > r.prev_slow and r.fast_ema <= r.slow_ema))

            if asset == 0 and crossup and cond_above and slow_trending_up:
                entry_price = Decimal(str(r['close'])) * (Decimal('1') + FEE)
                asset = balance / entry_price
                balance = Decimal('0')

            elif asset > 0:
                tp_price = entry_price * (Decimal('1') + tp_value + FEE + FEE)
                if r['high'] >= float(tp_price):
                    exit_price = tp_price * (Decimal('1') - FEE)
                    balance = asset * exit_price
                    trades += 1
                    wins += int(exit_price > entry_price)
                    asset = Decimal('0')
                    continue
                if crossdown:
                    exit_price = Decimal(str(r['close'])) * (Decimal('1') - FEE)
                    balance = asset * exit_price
                    trades += 1
                    wins += int(exit_price > entry_price)
                    asset = Decimal('0')

        if asset > 0:
            final_price = Decimal(str(data['close'].iloc[-1])) * (Decimal('1') - FEE)
            balance = asset * final_price
            trades += 1
            wins += int(final_price > entry_price)

        pnl = balance - Decimal('1000')
        pnl_pct = (balance / Decimal('1000') - Decimal('1')) * Decimal('100')
        return float(pnl), float(pnl_pct), trades, wins

    def find_best_params(self, df, trials):
        results = []
        for i in range(trials):
            if not self._running:
                break
            f = 5
            s = 155
            m = random.randint(8, 150)
            tp_rand = Decimal(str(round(random.uniform(0.0033, 0.0070), 5)))
            pnl, pnl_pct, trades, wins = self.triple_ema_backtest(df, f, m, s, tp_rand)
            winrate = (wins / trades) if trades > 0 else 0
            if trades > 0 and pnl > 0 and wins > 0:
                results.append({
                    'fast': f, 'mid': m, 'slow': s, 'tp': float(tp_rand),
                    'pnl': pnl, 'pnl_pct': pnl_pct,
                    'trades': trades, 'wins': wins, 'winrate': winrate
                })
            self.progress_signal.emit(int((i / trials) * 100))
        if not results:
            return None
        df_res = pd.DataFrame(results).sort_values(by=['winrate','pnl'], ascending=[False,False]).reset_index(drop=True)
        best = df_res.iloc[0]
        self.log(f"🔥 Best Backtest → Fast={best.fast}, Mid={best.mid}, Slow={best.slow}, TP={best.tp*100:.2f}%, "
                 f"Winrate={best.winrate*100:.2f}%, Trades={int(best.trades)}, Wins={int(best.wins)}, "
                 f"PnL={best.pnl:.2f} USDT ({best.pnl_pct:.2f}%)")
        return int(best.fast), int(best.mid), int(best.slow), Decimal(str(best.tp))

    def reoptimize(self, trials):
        self.log("[INFO] Re-optimizing after sell...")
        df = self.fetch_ohlcv(self.lookback, extra_candles=200)
        best = self.find_best_params(df, trials)
        if best:
            self.fast, self.mid, self.slow, self.tp_val = best
            self.log(f"[INFO] Updated strategy → Fast={self.fast}, Mid={self.mid}, Slow={self.slow}, TP={self.tp_val*100:.2f}%")
        else:
            self.log("[WARN] Re-optimization failed. Keeping previous params.")

    async def minute_logger(self):
        while self._running:
            try:
                price = float(self.client.ticker_price(self.symbol)['price'])
                p_f, p_m, p_s = self.ema_fast, self.ema_mid, self.ema_slow
                self.ema_fast = self.next_ema(price, self.ema_fast, self.fast)
                self.ema_mid = self.next_ema(price, self.ema_mid, self.mid)
                self.ema_slow = self.next_ema(price, self.ema_slow, self.slow)
                ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
                self.log(f"[{ts}] 1-min EMA update | Fast: {self.ema_fast:.4f} Mid: {self.ema_mid:.4f} Slow: {self.ema_slow:.4f}")
            except Exception as e:
                self.log(f"[WARN] EMA log error: {e}")
            await asyncio.sleep(60)

    def next_ema(self, price, prev, span):
        alpha = Decimal('2') / (Decimal(str(span)) + Decimal('1'))
        price = Decimal(str(price))
        prev = Decimal(str(prev))
        return float((price - prev) * alpha + prev)

    async def websocket_strategy_loop(self):
        uri = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.interval}"
        in_position = False
        entry_price = None

        while self._running:
            try:
                async with websockets.connect(uri, ping_interval=20) as ws:
                    async for msg in ws:
                        if not self._running:
                            break
                        k = json.loads(msg)['k']
                        if not k['x']:  # only on candle close
                            continue
                        price = float(k['c'])
                        high = float(k['h'])
                        open_ = float(k['o'])
                        close = price

                        p_f, p_m, p_s = self.ema_fast, self.ema_mid, self.ema_slow
                        self.ema_fast = self.next_ema(close, self.ema_fast, self.fast)
                        self.ema_mid = self.next_ema(close, self.ema_mid, self.mid)
                        self.ema_slow = self.next_ema(close, self.ema_slow, self.slow)

                        ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
                        self.log(f"[{ts}] CANDLE CLOSE | Fast:{self.ema_fast:.4f} Mid:{self.ema_mid:.4f} Slow:{self.ema_slow:.4f}")

                        crossup = (p_f < p_m and self.ema_fast >= self.ema_mid)
                        slow_trending_up = self.ema_slow >= p_s  # 🆕 filter
                        cond_above = self.ema_mid > self.ema_slow or open_ > self.ema_slow or close > self.ema_slow
                        crossdown = ((p_f > p_m and self.ema_fast <= self.ema_mid) or (p_f > p_s and self.ema_fast <= self.ema_slow))

                        if not in_position and crossup and cond_above and slow_trending_up:
                            usdt_bal = await self.async_balance('USDT')
                            qty = self.calc_buy_qty(usdt_bal, price)
                            if qty:
                                try:
                                    self.client.new_order(symbol=self.symbol, side='BUY', type='MARKET', quantity=qty)
                                    in_position = True
                                    entry_price = Decimal(str(price))
                                    self.log(f"[BUY] @ {price:.6f} qty={qty}")
                                except Exception as e:
                                    self.log(f"[ERROR] BUY FAILED: {e}")

                        if in_position:
                            tp_price = entry_price * (Decimal('1') + self.tp_val + Decimal('0.001')*2)
                            if Decimal(str(high)) >= tp_price:
                                asset_bal = await self.async_balance('BNB')
                                qty = self.calc_sell_qty(asset_bal, price)
                                if qty:
                                    try:
                                        self.client.new_order(symbol=self.symbol, side='SELL', type='MARKET', quantity=qty)
                                        in_position = False
                                        entry_price = None
                                        self.log(f"[TP SELL] @ {price:.6f}")
                                        self.reoptimize(self.reopt_trials)
                                    except Exception as e:
                                        self.log(f"[ERROR] TP SELL FAILED: {e}")
                            elif crossdown:
                                asset_bal = await self.async_balance('BNB')
                                qty = self.calc_sell_qty(asset_bal, price)
                                if qty:
                                    try:
                                        self.client.new_order(symbol=self.symbol, side='SELL', type='MARKET', quantity=qty)
                                        in_position = False
                                        entry_price = None
                                        self.log(f"[CROSSDOWN SELL] @ {price:.6f}")
                                        self.reoptimize(self.reopt_trials)
                                    except Exception as e:
                                        self.log(f"[ERROR] CROSSDOWN SELL FAILED: {e}")
            except Exception as e:
                self.log(f"⚠️ WS Error {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    async def main_logic(self):
        self.client = Spot(api_key=self.cfg['api_key'], api_secret=self.cfg['api_secret'])
        self.symbol = self.cfg['symbol'].upper()
        self.interval = self.cfg['interval']
        self.lookback = int(self.cfg['lookback'])
        self.initial_trials = int(self.cfg['initial_trials'])
        self.reopt_trials = int(self.cfg['reopt_trials'])

        self.get_symbol_info()
        self.log(f"[INFO] Fetching {self.symbol} OHLCV data...")
        df = self.fetch_ohlcv(self.lookback, extra_candles=200)
        self.log(f"[INFO] Optimizing strategy over {self.initial_trials} trials...")
        best = self.find_best_params(df, self.initial_trials)
        if not best:
            self.log("[⚠️] No profitable strategy found. Exiting.")
            return
        self.fast, self.mid, self.slow, self.tp_val = best

        self.ema_fast = df['ohlc4'].ewm(span=self.fast, adjust=False).mean().iloc[-1]
        self.ema_mid  = df['ohlc4'].ewm(span=self.mid, adjust=False).mean().iloc[-1]
        self.ema_slow = df['ohlc4'].ewm(span=self.slow, adjust=False).mean().iloc[-1]

        self.log(f"[INFO] Starting live trading with TP={self.tp_val*100:.2f}%")

        task1 = asyncio.create_task(self.websocket_strategy_loop())
        task2 = asyncio.create_task(self.minute_logger())
        await asyncio.gather(task1, task2)

# ------------------- GUI ------------------- #
class EMABotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📊 Triple EMA Trading Bot")
        self.setMinimumWidth(720)
        self.thread = None

        self.api_key_input = QLineEdit()
        self.api_secret_input = QLineEdit()
        self.api_secret_input.setEchoMode(QLineEdit.Password)
        self.symbol_input = QLineEdit("bnbusdt")
        self.interval_input = QLineEdit("15m")
        self.lookback_input = QLineEdit("72")
        self.initial_trials_input = QLineEdit("2000")
        self.reopt_trials_input = QLineEdit("500")

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.progress = QProgressBar()

        self.save_btn = QPushButton("💾 Save Config")
        self.load_btn = QPushButton("📂 Load Config")
        self.clear_btn = QPushButton("🧹 Clear Config")
        self.start_btn = QPushButton("▶️ Start")
        self.stop_btn = QPushButton("⏹️ Stop")

        self.layout_ui()
        self.connect_signals()
        self.load_config_if_exists()

    def layout_ui(self):
        w = QWidget()
        v = QVBoxLayout(w)

        def row(label, widget):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(widget)
            v.addLayout(h)

        row("API Key:", self.api_key_input)
        row("API Secret:", self.api_secret_input)
        row("Symbol:", self.symbol_input)
        row("Interval:", self.interval_input)
        row("Lookback Hours:", self.lookback_input)
        row("Initial Trials:", self.initial_trials_input)
        row("Re-Opt Trials:", self.reopt_trials_input)

        h_btn = QHBoxLayout()
        h_btn.addWidget(self.save_btn)
        h_btn.addWidget(self.load_btn)
        h_btn.addWidget(self.clear_btn)
        h_btn.addWidget(self.start_btn)
        h_btn.addWidget(self.stop_btn)
        v.addLayout(h_btn)

        v.addWidget(self.progress)
        v.addWidget(QLabel("Console:"))
        v.addWidget(self.console)
        self.setCentralWidget(w)

    def connect_signals(self):
        self.save_btn.clicked.connect(self.save_config)
        self.load_btn.clicked.connect(self.load_config_dialog)
        self.clear_btn.clicked.connect(self.clear_config)
        self.start_btn.clicked.connect(self.start_trading)
        self.stop_btn.clicked.connect(self.stop_trading)

    def get_config(self):
        return {
            "api_key": self.api_key_input.text().strip(),
            "api_secret": self.api_secret_input.text().strip(),
            "symbol": self.symbol_input.text().strip(),
            "interval": self.interval_input.text().strip(),
            "lookback": self.lookback_input.text().strip(),
            "initial_trials": self.initial_trials_input.text().strip(),
            "reopt_trials": self.reopt_trials_input.text().strip()
        }

    def save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.get_config(), f, indent=2)
        self.console.appendPlainText("[INFO] Config saved.")

    def load_config_if_exists(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE) as f:
                self.populate_fields(json.load(f))
            self.console.appendPlainText("[INFO] Config loaded from file.")

    def load_config_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON Files (*.json)")
        if path:
            with open(path) as f:
                self.populate_fields(json.load(f))
            self.console.appendPlainText(f"[INFO] Loaded config from {path}")

    def populate_fields(self, cfg):
        self.api_key_input.setText(cfg.get("api_key",""))
        self.api_secret_input.setText(cfg.get("api_secret",""))
        self.symbol_input.setText(cfg.get("symbol",""))
        self.interval_input.setText(cfg.get("interval",""))
        self.lookback_input.setText(cfg.get("lookback",""))
        self.initial_trials_input.setText(cfg.get("initial_trials",""))
        self.reopt_trials_input.setText(cfg.get("reopt_trials",""))

    def clear_config(self):
        if os.path.exists(CONFIG_FILE):
            os.remove(CONFIG_FILE)
        self.console.appendPlainText("[INFO] Config cleared.")

    def start_trading(self):
        cfg = self.get_config()
        self.thread = TradingThread(cfg)
        self.thread.log_signal.connect(self.console.appendPlainText)
        self.thread.progress_signal.connect(self.progress.setValue)
        self.thread.finished_signal.connect(self.console.appendPlainText)
        self.thread.start()

    def stop_trading(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.console.appendPlainText("[INFO] Stop signal sent.")

# ------------------- Main ------------------- #
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = EMABotGUI()
    gui.show()
    sys.exit(app.exec())
