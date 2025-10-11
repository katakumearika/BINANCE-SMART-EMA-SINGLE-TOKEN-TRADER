#Trading logic updated
import sys
import warnings
import asyncio
import json
import pandas as pd
from binance.spot import Spot
import websockets
from datetime import datetime, timezone
from decimal import Decimal, getcontext
import concurrent.futures
import random
import multiprocessing

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QPlainTextEdit, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QDoubleValidator

getcontext().prec = 16

FEE_RATE         = Decimal('0.001')
COOLDOWN_MINUTES = 6
TP_REOPTIMIZE_HRS = 18
DEFAULT_PAIR     = 'BNBUSDT'
DEFAULT_LOOKBACK = 5 # days
DEFAULT_NTRIALS  = 50000

# --- Output Filter for Frozen EXE to hide debugger warnings ---
def filter_debugger_warnings():
    class DebuggerWarningFilter:
        def write(self, msg):
            if "Debugger warning:" in msg or "frozen modules" in msg:
                return
            sys.__stderr__.write(msg)
        def flush(self):
            sys.__stderr__.flush()
    sys.stderr = DebuggerWarningFilter()

if getattr(sys, 'frozen', False):
    filter_debugger_warnings()
# -------------------------------------------------------------

client = None
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def trend_str(curr, prev):
    if curr > prev:
        return "UP"
    elif curr < prev:
        return "DOWN"
    return "FLAT"

def next_ema(price, prev_ema, span):
    alpha = Decimal('2') / (Decimal(str(span)) + Decimal('1'))
    return float((Decimal(str(price)) - Decimal(str(prev_ema))) * alpha + Decimal(str(prev_ema)))

def load_config():
    try:
        with open('config.json','r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(cfg):
    with open('config.json','w') as f:
        json.dump(cfg, f)

def init_client(api_key, api_secret):
    global client
    client = Spot(api_key=api_key, api_secret=api_secret)

def get_symbol_info(symbol):
    info = client.exchange_info(symbol=symbol)
    filters = {f['filterType']: f for f in info['symbols'][0]['filters']}
    min_qty = Decimal(filters['LOT_SIZE']['minQty'])
    step_size = Decimal(filters['LOT_SIZE']['stepSize'])
    min_notional = Decimal(filters['MIN_NOTIONAL']['minNotional']) if 'MIN_NOTIONAL' in filters else None
    return min_qty, step_size, min_notional

def get_price(symbol):
    ticker = client.ticker_price(symbol=symbol)
    return Decimal(str(ticker['price']))

def get_balance(asset):
    acc = client.account()
    for b in acc['balances']:
        if b['asset'] == asset:
            return Decimal(str(b['free']))
    return Decimal('0')

async def async_balance(asset):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: get_balance(asset))

async def async_new_order(**kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: client.new_order(**kwargs))

def fetch_ohlcv(symbol, interval, hours):
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - int(float(hours) * 3600 * 1000)
    klines = client.klines(
        symbol=symbol,
        interval=interval,
        startTime=start_ms,
        endTime=end_ms,
        limit=1000
    )
    df = pd.DataFrame(klines, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','quote_vol','n_trades','taker_bv','taker_qv','ignore'
    ])
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    return df

def multi_ema_backtest_worker(args):
    symbol, df, fast_span, mid_span, slow_span, tp_net = args
    d = df.copy()
    d['fast_ema'] = d['ohlc4'].ewm(span=fast_span, adjust=False).mean()
    d['mid_ema']  = d['ohlc4'].ewm(span=mid_span,  adjust=False).mean()
    d['slow_ema'] = d['ohlc4'].ewm(span=slow_span, adjust=False).mean()
    d['prev_fast'] = d['fast_ema'].shift(1)
    d['prev_mid']  = d['mid_ema'].shift(1)
    d['prev_slow'] = d['slow_ema'].shift(1)

    balance = Decimal('1000')
    asset = Decimal('0')
    entry_price = None
    trades = 0
    wins = 0
    hist = []
    mid_hist = []

    for r in d.itertuples():
        price = Decimal(str(r.close))
        hist.append(r.slow_ema)
        mid_hist.append(r.mid_ema)
        if len(hist) > COOLDOWN_MINUTES + 1:
            hist.pop(0)
        if len(mid_hist) > 2:
            mid_hist.pop(0)
        # ENTRY
        if asset == 0 and len(hist) == COOLDOWN_MINUTES+1 and len(mid_hist) == 2:
            fast_cross_up = (r.prev_fast < r.prev_mid) and (r.fast_ema > r.mid_ema)
            fast_above_slow = r.fast_ema > r.slow_ema
            mid_above_slow = r.mid_ema > r.slow_ema
            if fast_cross_up and fast_above_slow and mid_above_slow:
                entry_price = price * (Decimal('1') + FEE_RATE)
                if entry_price is not None and entry_price > Decimal('0'):
                    try:
                        asset = (balance / entry_price).quantize(Decimal('1e-8'))
                        balance = Decimal('0')
                    except Exception:
                        asset = Decimal('0')
                        entry_price = None
                else:
                    asset = Decimal('0')
                    entry_price = None
        # EXIT
        elif asset > 0 and entry_price is not None and entry_price > Decimal('0'):
            tp_price = entry_price * (Decimal('1') + tp_net)
            fast_cross_down = (r.prev_fast > r.prev_mid) and (r.fast_ema < r.mid_ema)
            fast_below_slow = (r.prev_fast > r.prev_slow) and (r.fast_ema < r.slow_ema)
            mid_below_slow  = (r.prev_mid > r.prev_slow) and (r.mid_ema < r.slow_ema)
            if price >= tp_price or fast_cross_down or fast_below_slow or mid_below_slow:
                exit_price = price * (Decimal('1') - FEE_RATE)
                balance = (asset * exit_price).quantize(Decimal('1e-8'))
                trades += 1
                if price >= entry_price:
                    wins += 1
                asset = Decimal('0')
                entry_price = None
    if asset > 0 and entry_price is not None and entry_price > Decimal('0'):
        final_price = Decimal(str(d['close'].iloc[-1])) * (Decimal('1') - FEE_RATE)
        balance = (asset * final_price).quantize(Decimal('1e-8'))
        trades += 1
        if final_price > entry_price:
            wins += 1
    pnl = float(balance - Decimal('1000'))
    winrate = (wins / trades) if trades > 0 else 0
    return {
        'symbol': symbol,
        'fast': fast_span,
        'mid': mid_span,
        'slow': slow_span,
        'tp': tp_net,
        'pnl': pnl,
        'winrate': winrate,
        'trades': trades
    }

def optimizer_single_symbol(symbol, lookback_days, tp_min=1.0, tp_max=2.0, n_trials=50000, progress_callback=None, log_callback=None):
    if log_callback: log_callback(f"Downloading OHLCV for {symbol} ...")
    df = fetch_ohlcv(symbol, '15m', float(lookback_days) * 24)
    combos = []
    min_tp = Decimal(str(tp_min)) / Decimal('100')
    max_tp = Decimal(str(tp_max)) / Decimal('100')
    for _ in range(n_trials):
        fast = random.randint(2, 15)
        mid  = random.randint(16, 50)
        slow = random.randint(80, 180)
        if not (fast < mid < slow):
            continue
        tp_raw = Decimal(str(round(random.uniform(float(min_tp), float(max_tp)), 4)))
        combos.append( (symbol, df, fast, mid, slow, tp_raw) )
    total = len(combos)
    if log_callback: log_callback(f"Backtesting {total} random strategy combos ...")
    results = []
    completed = 0
    with multiprocessing.get_context('spawn').Pool(processes=multiprocessing.cpu_count()) as pool:
        for res in pool.imap_unordered(multi_ema_backtest_worker, combos, chunksize=6):
            results.append(res)
            completed += 1
            if progress_callback:
                percent = int(completed / total * 100)
                progress_callback(percent)
    # --- Highest winrate (then most trades) ---
    best = max(results, key=lambda x: (x['winrate'], x['trades']))
    return (best['fast'], best['mid'], best['slow'], best['tp']), best

async def live_loop_token(symbol, fast_span, mid_span, slow_span, tp_net, log_callback=None, reoptimize_timer=TP_REOPTIMIZE_HRS*3600):
    min_qty, step_size, min_notional = get_symbol_info(symbol)
    in_position = False
    fast_ema = None
    mid_ema  = None
    slow_ema = None
    hist = []
    mid_hist = []
    entry_price = None
    uri = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_1m"
    start_time = datetime.now()
    prev_fast = prev_mid = prev_slow = None
    async with websockets.connect(uri, ping_interval=20) as ws:
        async for msg in ws:
            if (datetime.now() - start_time).total_seconds() > reoptimize_timer:
                if log_callback: log_callback("[TIMER] 18 hours elapsed; re-optimizing...")
                return
            k = json.loads(msg)['k']
            if not k['x']:
                continue
            o,h,l,c = map(float,(k['o'],k['h'],k['l'],k['c']))
            price = Decimal(str(c))
            ohlc4 = (o+h+l+c)/4
            fast_ema = ohlc4 if fast_ema is None else next_ema(ohlc4, fast_ema, fast_span)
            mid_ema  = ohlc4 if mid_ema  is None else next_ema(ohlc4, mid_ema,  mid_span)
            slow_ema = ohlc4 if slow_ema is None else next_ema(ohlc4, slow_ema, slow_span)
            hist.append(slow_ema)
            mid_hist.append(mid_ema)
            if len(hist)>COOLDOWN_MINUTES+1:
                hist.pop(0)
            if len(mid_hist)>2:
                mid_hist.pop(0)
            ts=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            fast_prev = prev_fast if prev_fast is not None else fast_ema
            mid_prev  = prev_mid  if prev_mid  is not None else mid_ema
            slow_prev = prev_slow if prev_slow is not None else slow_ema
            msglog = f"{ts} {symbol} SLOW={slow_ema:.6f}({trend_str(slow_ema,slow_prev)}) FAST={fast_ema:.6f}({trend_str(fast_ema,fast_prev)}) MID={mid_ema:.6f}({trend_str(mid_ema,mid_prev)})"
            if log_callback: log_callback(msglog)
            else: print(msglog)
            # ENTRY
            if (not in_position and len(hist) == COOLDOWN_MINUTES+1 and len(mid_hist) == 2):
                fast_cross_up = (fast_prev < mid_prev) and (fast_ema > mid_ema)
                fast_above_slow = fast_ema > slow_ema
                mid_above_slow = mid_ema > slow_ema
                if fast_cross_up and fast_above_slow and mid_above_slow:
                    bal=await async_balance(symbol[:-4])
                    if bal*price<=Decimal('5'):
                        ub=await async_balance('USDT')
                        qty=float((ub*(Decimal('1')-FEE_RATE)/price)//step_size*step_size)
                        if Decimal(str(qty))>=min_qty:
                            entry_price=price*(Decimal('1')+FEE_RATE)
                            entrymsg = f"{ts} BUY {symbol}@{c:.6f} qty={qty}"
                            if log_callback: log_callback(entrymsg)
                            else: print(entrymsg)
                            await async_new_order(symbol=symbol,side='BUY',type='MARKET',quantity=qty)
                            in_position=True
            # EXIT
            if in_position:
                tp_price = entry_price * (Decimal('1') + tp_net)
                fast_cross_down = (fast_prev > mid_prev) and (fast_ema < mid_ema)
                fast_below_slow = (fast_prev > slow_prev) and (fast_ema < slow_ema)
                mid_below_slow = (mid_prev > slow_prev) and (mid_ema < slow_ema)
                if price >= tp_price or fast_cross_down or fast_below_slow or mid_below_slow:
                    qty = float((await async_balance(symbol[:-4])//step_size)*step_size)
                    exitmsg = f"{ts} SELL {symbol}@{c:.6f} qty={qty} {'[TP]' if price>=tp_price else '[EMA cross-down]'}"
                    if log_callback: log_callback(exitmsg)
                    else: print(exitmsg)
                    await async_new_order(symbol=symbol,side='SELL',type='MARKET',quantity=qty)
                    return
            prev_fast = fast_ema
            prev_mid = mid_ema
            prev_slow = slow_ema
class ContinuousTradingWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal()
    def __init__(self, get_symbol_func, get_tp_range_func, get_lookback_func, get_ntrials_func):
        super().__init__()
        self.get_symbol_func = get_symbol_func
        self.get_tp_range_func = get_tp_range_func
        self.get_lookback_func = get_lookback_func
        self.get_ntrials_func = get_ntrials_func
        self._abort = False
    def run(self):
        while not self._abort:
            try:
                symbol = self.get_symbol_func().strip().upper()
                tp_min, tp_max = self.get_tp_range_func()
                lookback = self.get_lookback_func()
                n_trials = self.get_ntrials_func()
                self.log.emit(f"=== Starting full re-optimization for {symbol} (lookback {lookback}d, TP {tp_min:.2f}% - {tp_max:.2f}%, {n_trials} tests) ===")
                best_params, stats = optimizer_single_symbol(
                    symbol,
                    lookback,
                    tp_min=tp_min,
                    tp_max=tp_max,
                    n_trials=n_trials,
                    progress_callback=self.progress.emit,
                    log_callback=self.log.emit
                )
                if best_params is None or stats is None:
                    self.log.emit("No result from optimizer, exiting.")
                    break
                f, m, s, tp = best_params
                pnl = stats['pnl']
                winrate = stats['winrate']
                trades = stats['trades']
                self.log.emit(f"Best: {symbol}, fast={f}, mid={m}, slow={s}, tp={float(tp)*100:.2f}%, PnL={pnl:.2f}, WR={winrate:.2%}, Trades={trades}")
                self.log.emit(f"--- Live trading {symbol} (fast={f}, mid={m}, slow={s}, tp={float(tp)*100:.2f}%) ---")
                asyncio.run(live_loop_token(symbol, f, m, s, tp, log_callback=self.log.emit))
                self.log.emit("Position closed or timer elapsed; re-optimizing...")
            except Exception as ex:
                self.log.emit(f"Live trading error: {ex}")
    def abort(self):
        self._abort = True

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BINANCE SMART-EMA SINGLE TOKEN TRADER")
        cfg = load_config()
        self.api_key = cfg.get('api_key','')
        self.api_secret = cfg.get('api_secret','')
        self.symbol_val = cfg.get('symbol', DEFAULT_PAIR)
        self.tp_min_val = str(cfg.get('tp_min', '1.0'))
        self.tp_max_val = str(cfg.get('tp_max', '2.0'))
        self.lookback_val = str(cfg.get('lookback', DEFAULT_LOOKBACK))
        self.ntrials_val = str(cfg.get('ntrials', str(DEFAULT_NTRIALS)))
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("API Key:"))
        self.key_in = QLineEdit(self.api_key)
        h1.addWidget(self.key_in)
        h1.addWidget(QLabel("Secret:"))
        self.sec_in = QLineEdit(self.api_secret)
        self.sec_in.setEchoMode(QLineEdit.Password)
        h1.addWidget(self.sec_in)
        layout.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Symbol:"))
        self.symbol_in = QLineEdit(self.symbol_val)
        h2.addWidget(self.symbol_in)
        h2.addWidget(QLabel("Lookback (days):"))
        self.lookback_in = QLineEdit(self.lookback_val)
        self.lookback_in.setValidator(QDoubleValidator(0.25, 60.0, 2, self))
        h2.addWidget(self.lookback_in)
        h2.addWidget(QLabel("Random Tests:"))
        self.ntrials_in = QLineEdit(self.ntrials_val)
        self.ntrials_in.setValidator(QDoubleValidator(10, 50000, 0, self))
        h2.addWidget(self.ntrials_in)
        layout.addLayout(h2)

        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Min TP (%):"))
        self.tp_min_in = QLineEdit(self.tp_min_val)
        self.tp_min_in.setValidator(QDoubleValidator(0.01, 10.0, 2, self))
        h3.addWidget(self.tp_min_in)
        h3.addWidget(QLabel("Max TP (%):"))
        self.tp_max_in = QLineEdit(self.tp_max_val)
        self.tp_max_in.setValidator(QDoubleValidator(0.01, 10.0, 2, self))
        h3.addWidget(self.tp_max_in)
        layout.addLayout(h3)

        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.clear_btn = QPushButton("Clear")
        self.reset_btn = QPushButton("Reset")
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.reset_btn)
        layout.addLayout(btn_layout)

        action_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Live Trading")
        self.exit_btn = QPushButton("Exit")
        action_layout.addWidget(self.start_btn)
        action_layout.addWidget(self.exit_btn)
        layout.addLayout(action_layout)

        self.progress = QProgressBar()
        self.progress.setRange(0,100)
        layout.addWidget(self.progress)
        self.result_lbl = QLabel("Best Params: None")
        layout.addWidget(self.result_lbl)
        self.log_out = QPlainTextEdit()
        self.log_out.setReadOnly(True)
        layout.addWidget(self.log_out)
        self.save_btn.clicked.connect(self.save_config)
        self.clear_btn.clicked.connect(self.clear_inputs)
        self.reset_btn.clicked.connect(self.reset_defaults)
        self.start_btn.clicked.connect(self.start_trading)
        self.exit_btn.clicked.connect(self.close)
        self.worker = None
        self.apply_api()

    def get_tp_range(self):
        try:
            min_tp = max(0.01, float(self.tp_min_in.text()))
            max_tp = min(10.0, float(self.tp_max_in.text()))
            if min_tp > max_tp:
                min_tp, max_tp = max_tp, min_tp
            return min_tp, max_tp
        except Exception:
            return 1.0, 2.0

    def get_symbol(self):
        return self.symbol_in.text().strip().upper() or DEFAULT_PAIR

    def get_lookback(self):
        try:
            v = float(self.lookback_in.text())
            return max(0.25, min(v, 60))
        except Exception:
            return DEFAULT_LOOKBACK

    def get_ntrials(self):
        try:
            v = int(float(self.ntrials_in.text()))
            return max(10, min(v, 50000))
        except Exception:
            return DEFAULT_NTRIALS

    def apply_api(self):
        k = self.key_in.text().strip()
        s = self.sec_in.text().strip()
        if k and s:
            try:
                init_client(k,s)
            except Exception as ex:
                self.log_out.appendPlainText(f"API init error: {ex}")

    def save_config(self):
        cfg = {
            'api_key': self.key_in.text(),
            'api_secret': self.sec_in.text(),
            'symbol': self.symbol_in.text().strip().upper(),
            'lookback': self.lookback_in.text(),
            'tp_min': self.tp_min_in.text(),
            'tp_max': self.tp_max_in.text(),
            'ntrials': self.ntrials_in.text()
        }
        save_config(cfg)
        self.log_out.appendPlainText("Config saved.")
        self.apply_api()

    def clear_inputs(self):
        self.key_in.clear()
        self.sec_in.clear()
        self.symbol_in.clear()
        self.lookback_in.clear()
        self.tp_min_in.clear()
        self.tp_max_in.clear()
        self.ntrials_in.clear()
        self.log_out.clear()
        self.progress.setValue(0)
        self.result_lbl.setText("Best Params: None")

    def reset_defaults(self):
        self.clear_inputs()
        self.symbol_in.setText(DEFAULT_PAIR)
        self.lookback_in.setText(str(DEFAULT_LOOKBACK))
        self.tp_min_in.setText('1.0')
        self.tp_max_in.setText('2.0')
        self.ntrials_in.setText(str(DEFAULT_NTRIALS))

    def start_trading(self):
        self.log_out.appendPlainText("Starting continuous optimization + live trading ...")
        self.apply_api()
        self.worker = ContinuousTradingWorker(self.get_symbol, self.get_tp_range, self.get_lookback, self.get_ntrials)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.handle_log)
        self.worker.finished.connect(lambda: self.log_out.appendPlainText("Trading finished."))
        self.worker.start()

    def handle_log(self, msg):
        self.log_out.appendPlainText(msg)
        if msg.startswith("Best:"):
            self.result_lbl.setText(msg)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec())
