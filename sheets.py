"""
sheets.py — Google Sheets Foundation Module
NEPSE AI Engine — Phase 2

The foundation of the entire system.
Every module reads/writes through here.
"""

import gspread
import os
import time
import logging
from datetime import datetime
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SHEETS] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

SHEET_ID        = os.getenv("GOOGLE_SHEETS_ID")
SERVICE_ACCOUNT = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH", "service_account.json")

# Tab names — must match your Google Sheet exactly
TABS = {
    "watchlist":          "WATCHLIST",
    "portfolio":          "PORTFOLIO",
    "market_log":         "MARKET_LOG",
    "learning_hub":       "LEARNING_HUB",
    "macro_data":         "MACRO_DATA",
    "geo_data":           "GEOPOLITICAL_DATA",
    "candle_patterns":    "CANDLE_PATTERNS",
    "capital_allocation": "CAPITAL_ALLOCATION",
    "settings":           "SETTINGS",
    "financials":         "FINANCIALS",
    "fundamentals":       "FUNDAMENTALS",
    "sector_momentum":    "SECTOR_MOMENTUM",
    "news_sentiment":     "NEWS_SENTIMENT",
    "corporate_events":   "CORPORATE_EVENTS",
    "market_breadth":     "MARKET_BREADTH",
    "financial_advisor":  "FINANCIAL_ADVISOR",
    "backtest_results":   "BACKTEST_RESULTS",
}

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


# ─────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────
_client     = None
_spreadsheet = None

def _get_client():
    """Get or create authenticated gspread client."""
    global _client
    if _client is None:
        try:
            creds = Credentials.from_service_account_file(
                SERVICE_ACCOUNT,
                scopes=SCOPES
            )
            _client = gspread.authorize(creds)
            log.info("Google Sheets client authenticated")
        except Exception as e:
            log.error(f"Authentication failed: {e}")
            raise
    return _client


def _get_sheet():
    """Get or create spreadsheet connection."""
    global _spreadsheet
    if _spreadsheet is None:
        try:
            client = _get_client()
            _spreadsheet = client.open_by_key(SHEET_ID)
            log.info(f"Connected to spreadsheet: {_spreadsheet.title}")
        except Exception as e:
            log.error(f"Failed to open spreadsheet: {e}")
            raise
    return _spreadsheet


def _get_tab(tab_name: str):
    """Get a worksheet by name with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            sheet = _get_sheet()
            worksheet = sheet.worksheet(tab_name)
            return worksheet
        except gspread.exceptions.WorksheetNotFound:
            log.error(f"Tab not found: {tab_name}")
            raise
        except gspread.exceptions.APIError as e:
            if attempt < MAX_RETRIES - 1:
                log.warning(f"API error on attempt {attempt+1}, retrying in {RETRY_DELAY}s: {e}")
                time.sleep(RETRY_DELAY)
            else:
                log.error(f"API error after {MAX_RETRIES} attempts: {e}")
                raise
        except Exception as e:
            log.error(f"Unexpected error getting tab {tab_name}: {e}")
            raise


# ─────────────────────────────────────────
# READ FUNCTIONS
# ─────────────────────────────────────────
def read_tab(tab_key: str) -> list[dict]:
    """
    Read all rows from a tab as list of dicts.
    
    Args:
        tab_key: Key from TABS dict (e.g. "settings")
    
    Returns:
        List of dicts with column headers as keys
    
    Example:
        settings = read_tab("settings")
        for row in settings:
            print(row["Key"], row["Value"])
    """
    tab_name = TABS.get(tab_key, tab_key)
    try:
        worksheet = _get_tab(tab_name)
        records = worksheet.get_all_records()
        log.info(f"Read {len(records)} rows from {tab_name}")
        return records
    except Exception as e:
        log.error(f"Failed to read tab {tab_name}: {e}")
        return []


def read_tab_raw(tab_key: str) -> list[list]:
    """
    Read all rows as raw list of lists (no header processing).
    Useful when headers have duplicates or special characters.
    """
    tab_name = TABS.get(tab_key, tab_key)
    try:
        worksheet = _get_tab(tab_name)
        values = worksheet.get_all_values()
        log.info(f"Read {len(values)} raw rows from {tab_name}")
        return values
    except Exception as e:
        log.error(f"Failed to read raw tab {tab_name}: {e}")
        return []


def get_setting(property_name: str, default=None):
    """
    Get a single setting value from SETTINGS tab based on the 'Property' column.
    
    SETTINGS tab structure:
    | Property | Value | Key | Description | Last_Updated | Set_By |
    
    Args:
        property_name: The name of the property to find
        default: Value to return if property not found
    
    Returns:
        Setting value as string, or default
    """
    try:
        settings = read_tab("settings")
        for row in settings:
            # Changed lookup from row.get("Key") to row.get("Property")
            if str(row.get("Property", "")).strip() == property_name:
                value = row.get("Value", default)
                log.info(f"Setting Property [{property_name}] = {value}")
                return value
        
        log.warning(f"Property [{property_name}] not found, using default: {default}")
        return default
    except Exception as e:
        log.error(f"Failed to get setting for property {property_name}: {e}")
        return default


def get_all_settings() -> dict:
    """
    Get all settings as a single dict.
    
    Returns:
        Dict of {key: value} for all settings
    
    Example:
        settings = get_all_settings()
        rsi = settings.get("Min_RSI_Entry", "28")
    """
    try:
        settings = read_tab("settings")
        result = {}
        for row in settings:
            key = str(row.get("Key", "")).strip()
            val = row.get("Value", "")
            if key:
                result[key] = val
        log.info(f"Loaded {len(result)} settings")
        return result
    except Exception as e:
        log.error(f"Failed to get all settings: {e}")
        return {}


def get_watchlist() -> list[str]:
    """
    Get list of stock symbols from WATCHLIST tab.
    
    Returns:
        List of symbol strings
    
    Example:
        symbols = get_watchlist()
        # ["NABIL", "HBL", "HIDCL", ...]
    """
    try:
        rows = read_tab("watchlist")
        symbols = [
            str(row.get("Symbol", "")).strip().upper()
            for row in rows
            if row.get("Symbol", "")
        ]
        log.info(f"Loaded {len(symbols)} symbols from watchlist")
        return symbols
    except Exception as e:
        log.error(f"Failed to get watchlist: {e}")
        return []


def get_open_positions() -> list[dict]:
    """
    Get all open positions from PORTFOLIO tab.
    Returns rows where Status = OPEN or Exit_Date is empty.
    """
    try:
        rows = read_tab("portfolio")
        open_pos = [
            row for row in rows
            if str(row.get("Status", "")).upper() == "OPEN"
            or not row.get("Exit_Date", "")
        ]
        log.info(f"Found {len(open_pos)} open positions")
        return open_pos
    except Exception as e:
        log.error(f"Failed to get open positions: {e}")
        return []


def get_recent_lessons(n: int = 5) -> list[dict]:
    """
    Get the most recent N lessons from LEARNING_HUB.
    Used to inject into Claude prompts.
    
    Args:
        n: Number of recent lessons to return
    
    Returns:
        List of lesson dicts
    """
    try:
        rows = read_tab("learning_hub")
        # Return last N rows (most recent lessons)
        recent = rows[-n:] if len(rows) >= n else rows
        log.info(f"Loaded {len(recent)} recent lessons")
        return recent
    except Exception as e:
        log.error(f"Failed to get lessons: {e}")
        return []


def get_macro_data() -> dict:
    """
    Get current macro data from MACRO_DATA tab.
    Returns as flat dict for easy access.
    """
    try:
        rows = read_tab("macro_data")
        macro = {}
        for row in rows:
            key = str(row.get("Indicator", "")).strip()
            val = row.get("Value", "")
            if key:
                macro[key] = val
        log.info(f"Loaded {len(macro)} macro indicators")
        return macro
    except Exception as e:
        log.error(f"Failed to get macro data: {e}")
        return {}


def get_geo_score() -> dict:
    """
    Get latest geopolitical score from GEOPOLITICAL_DATA tab.
    Returns the most recent row.
    """
    try:
        rows = read_tab("geo_data")
        if not rows:
            return {"geo_score": 0, "status": "NEUTRAL"}
        latest = rows[-1]
        log.info(f"Geo score: {latest.get('Geo_Score', 0)}")
        return latest
    except Exception as e:
        log.error(f"Failed to get geo score: {e}")
        return {"geo_score": 0, "status": "NEUTRAL"}


# ─────────────────────────────────────────
# WRITE FUNCTIONS
# ─────────────────────────────────────────
def write_row(tab_key: str, row_data: dict) -> bool:
    """
    Append a new row to a tab.
    Automatically adds timestamp if not provided.
    
    Args:
        tab_key: Key from TABS dict
        row_data: Dict of {column_name: value}
    
    Returns:
        True if successful, False otherwise
    
    Example:
        write_row("market_log", {
            "Date": "2026-03-15",
            "Symbol": "NABIL",
            "Action": "BUY",
            "Entry": 850,
            "Confidence": 82
        })
    """
    tab_name = TABS.get(tab_key, tab_key)
    for attempt in range(MAX_RETRIES):
        try:
            worksheet = _get_tab(tab_name)

            # Get existing headers
            headers = worksheet.row_values(1)
            if not headers:
                log.error(f"No headers found in {tab_name}")
                return False

            # Build row in correct column order
            row = []
            for header in headers:
                value = row_data.get(header, "")
                # Auto-timestamp
                if header in ["Timestamp", "Created_At"] and not value:
                    value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                row.append(value)

            worksheet.append_row(row, value_input_option="USER_ENTERED")
            log.info(f"Row written to {tab_name}: {list(row_data.keys())}")
            return True

        except gspread.exceptions.APIError as e:
            if attempt < MAX_RETRIES - 1:
                log.warning(f"Retry {attempt+1} writing to {tab_name}: {e}")
                time.sleep(RETRY_DELAY)
            else:
                log.error(f"Failed to write row to {tab_name}: {e}")
                return False
        except Exception as e:
            log.error(f"Unexpected error writing to {tab_name}: {e}")
            return False

    return False


def update_cell(tab_key: str, row: int, col: int, value) -> bool:
    """
    Update a specific cell by row and column number.
    
    Args:
        tab_key: Key from TABS dict
        row: Row number (1-indexed, 1 = header)
        col: Column number (1-indexed)
        value: New value
    
    Returns:
        True if successful
    """
    tab_name = TABS.get(tab_key, tab_key)
    try:
        worksheet = _get_tab(tab_name)
        worksheet.update_cell(row, col, value)
        log.info(f"Updated cell ({row},{col}) in {tab_name} = {value}")
        return True
    except Exception as e:
        log.error(f"Failed to update cell in {tab_name}: {e}")
        return False


def update_cell_by_header(tab_key: str, row: int,
                           header: str, value) -> bool:
    """
    Update a cell by row number and column header name.
    Safer than using column numbers directly.
    
    Args:
        tab_key: Key from TABS dict
        row: Row number (2+ for data rows)
        header: Column header name
        value: New value
    
    Example:
        # Update outcome of trade in row 5
        update_cell_by_header("market_log", 5, "Outcome", "WIN")
    """
    tab_name = TABS.get(tab_key, tab_key)
    try:
        worksheet = _get_tab(tab_name)
        headers = worksheet.row_values(1)
        if header not in headers:
            log.error(f"Header '{header}' not found in {tab_name}")
            return False
        col = headers.index(header) + 1
        worksheet.update_cell(row, col, value)
        log.info(f"Updated [{tab_name}] row {row} [{header}] = {value}")
        return True
    except Exception as e:
        log.error(f"Failed to update cell by header in {tab_name}: {e}")
        return False


def update_setting(Prop: str, value, description: str = "") -> bool:
    """
    Update or create a setting in SETTINGS tab.
    
    Args:
        key: Setting key name
        value: New value
        description: Optional description update
    
    Returns:
        True if successful
    
    Example:
        update_setting("Market_State", "BEAR")
        update_setting("Min_RSI_Entry", 28, "Calibrated from backtest")
        update_setting("Win_Rate_30d", 67.5)
    """
    try:
        worksheet = _get_tab(TABS["settings"])
        data = worksheet.get_all_values()

        if not data:
            return False

        headers = data[0]
        key_col = headers.index("Property") + 1 if "Property" in headers else 1
        val_col = headers.index("Value") + 1 if "Value" in headers else 2
        upd_col = headers.index("Last_Updated") + 1 if "Last_Updated" in headers else None
        desc_col = headers.index("Description") + 1 if "Description" in headers else None

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Find existing row
        for i, row in enumerate(data[1:], start=2):
            if len(row) >= key_col and row[key_col - 1] == Prop:
                worksheet.update_cell(i, val_col, value)
                if upd_col:
                    worksheet.update_cell(i, upd_col, now)
                if desc_col and description:
                    worksheet.update_cell(i, desc_col, description)
                log.info(f"Updated setting [{Prop}] = {value}")
                return True

        # Key not found — create new row
        new_row = [""] * len(headers)
        new_row[key_col - 1] = Prop
        new_row[val_col - 1] = str(value)
        if upd_col:
            new_row[upd_col - 1] = now
        if desc_col and description:
            new_row[desc_col - 1] = description
        worksheet.append_row(new_row, value_input_option="USER_ENTERED")
        log.info(f"Created new setting [{Prop}] = {value}")
        return True

    except Exception as e:
        log.error(f"Failed to update setting [{Prop}]: {e}")
        return False


def update_kpi(kpi_name: str, value) -> bool:
    """
    Update a KPI value in FINANCIALS tab.
    Shortcut for update_setting on FINANCIALS.
    
    Args:
        kpi_name: KPI name matching FINANCIALS tab
        value: New KPI value
    
    Example:
        update_kpi("Win_Rate_30d", 67.5)
        update_kpi("Max_Drawdown_Pct", 4.2)
        update_kpi("Profit_Factor", 1.8)
    """
    try:
        worksheet = _get_tab(TABS["financials"])
        data = worksheet.get_all_values()

        if not data:
            return False

        headers = data[0]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Find KPI row
        kpi_col = headers.index("KPI_Name") + 1 if "KPI_Name" in headers else 1
        val_col = headers.index("Current_Value") + 1 if "Current_Value" in headers else 2
        upd_col = headers.index("Last_Updated") + 1 if "Last_Updated" in headers else None

        for i, row in enumerate(data[1:], start=2):
            if len(row) >= kpi_col and row[kpi_col - 1] == kpi_name:
                worksheet.update_cell(i, val_col, value)
                if upd_col:
                    worksheet.update_cell(i, upd_col, now)
                log.info(f"Updated KPI [{kpi_name}] = {value}")
                return True

        log.warning(f"KPI [{kpi_name}] not found in FINANCIALS")
        return False

    except Exception as e:
        log.error(f"Failed to update KPI [{kpi_name}]: {e}")
        return False


def write_lesson(lesson: dict) -> bool:
    """
    Write a new lesson to LEARNING_HUB tab.
    Called by auditor.py after each trade closes.
    
    Args:
        lesson: Dict with lesson details
    
    Example:
        write_lesson({
            "Date": "2026-03-15",
            "Symbol": "NABIL",
            "Pattern": "RSI_OVERBOUGHT_ENTRY",
            "Lesson": "RSI above 68 on banking = avoid",
            "Outcome": "LOSS",
            "PnL_NPR": -1610,
            "Confidence": "HIGH",
            "Applied_Count": 0
        })
    """
    return write_row("learning_hub", lesson)


def write_signal(signal: dict) -> bool:
    """
    Write a trade signal to MARKET_LOG tab.
    Called by main.py when signal is generated.
    """
    return write_row("market_log", signal)


def write_geo_update(geo_data: dict) -> bool:
    """
    Write geo sentiment update to GEOPOLITICAL_DATA tab.
    Called by geo_sentiment.py every 30 minutes.
    """
    return write_row("geo_data", geo_data)


def clear_tab(tab_key: str, keep_header: bool = True) -> bool:
    """
    Clear all data from a tab.
    
    Args:
        tab_key: Key from TABS dict
        keep_header: If True, preserves header row
    
    Returns:
        True if successful
    """
    tab_name = TABS.get(tab_key, tab_key)
    try:
        worksheet = _get_tab(tab_name)
        if keep_header:
            # Get header first
            header = worksheet.row_values(1)
            worksheet.clear()
            if header:
                worksheet.append_row(header)
            log.info(f"Cleared {tab_name} (header preserved)")
        else:
            worksheet.clear()
            log.info(f"Cleared {tab_name} completely")
        return True
    except Exception as e:
        log.error(f"Failed to clear {tab_name}: {e}")
        return False


def find_row(tab_key: str, column: str, value: str) -> int:
    """
    Find the row number of a specific value in a column.
    
    Args:
        tab_key: Key from TABS dict
        column: Column header name
        value: Value to search for
    
    Returns:
        Row number (1-indexed) or -1 if not found
    
    Example:
        row = find_row("market_log", "Symbol", "NABIL")
        row = find_row("portfolio", "Status", "OPEN")
    """
    tab_name = TABS.get(tab_key, tab_key)
    try:
        worksheet = _get_tab(tab_name)
        data = worksheet.get_all_values()

        if not data:
            return -1

        headers = data[0]
        if column not in headers:
            log.error(f"Column '{column}' not found in {tab_name}")
            return -1

        col_idx = headers.index(column)
        for i, row in enumerate(data[1:], start=2):
            if len(row) > col_idx and str(row[col_idx]).strip() == str(value).strip():
                return i

        return -1

    except Exception as e:
        log.error(f"Failed to find row in {tab_name}: {e}")
        return -1


def update_trade_outcome(symbol: str, entry_date: str,
                          outcome: str, actual_pnl: float,
                          exit_price: float) -> bool:
    """
    Update outcome of a trade in MARKET_LOG.
    Called by auditor.py when trade closes.
    
    Args:
        symbol: Stock symbol
        entry_date: Date of entry signal
        outcome: WIN / LOSS / BREAKEVEN
        actual_pnl: Actual profit/loss in NPR
        exit_price: Price at exit
    
    Returns:
        True if updated successfully
    """
    tab_name = TABS["market_log"]
    try:
        worksheet = _get_tab(tab_name)
        data = worksheet.get_all_values()

        if not data:
            return False

        headers = data[0]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for i, row in enumerate(data[1:], start=2):
            row_symbol = row[headers.index("Symbol")] if "Symbol" in headers else ""
            row_date   = row[headers.index("Date")] if "Date" in headers else ""
            row_outcome = row[headers.index("Outcome")] if "Outcome" in headers else ""

            if (row_symbol == symbol and
                row_date == entry_date and
                row_outcome == "PENDING"):

                if "Outcome" in headers:
                    worksheet.update_cell(i, headers.index("Outcome") + 1, outcome)
                if "Actual_PnL" in headers:
                    worksheet.update_cell(i, headers.index("Actual_PnL") + 1, actual_pnl)
                if "Exit_Price" in headers:
                    worksheet.update_cell(i, headers.index("Exit_Price") + 1, exit_price)
                if "Exit_Date" in headers:
                    worksheet.update_cell(i, headers.index("Exit_Date") + 1, now)

                log.info(f"Updated trade outcome: {symbol} {entry_date} = {outcome} {actual_pnl} NPR")
                return True

        log.warning(f"No PENDING trade found for {symbol} on {entry_date}")
        return False

    except Exception as e:
        log.error(f"Failed to update trade outcome: {e}")
        return False


# ─────────────────────────────────────────
# DYNAMIC SCHEMA FUNCTIONS
# ─────────────────────────────────────────

def tab_exists(tab_name: str) -> bool:
    """
    Check if a tab exists in the spreadsheet.

    Args:
        tab_name: Exact tab name (not key)

    Returns:
        True if exists, False otherwise

    Example:
        if not tab_exists("NEWS_SENTIMENT"):
            create_tab("NEWS_SENTIMENT", [...])
    """
    try:
        sheet = _get_sheet()
        existing = [ws.title for ws in sheet.worksheets()]
        return tab_name in existing
    except Exception as e:
        log.error(f"Failed to check tab existence: {e}")
        return False


def create_tab(tab_name: str, columns: list[str],
               register: bool = True) -> bool:
    """
    Dynamically create a new tab with headers.
    Registers in SCHEMA tab automatically.

    Args:
        tab_name: Name for the new tab
        columns: List of column header strings
        register: Whether to register in SCHEMA tab

    Returns:
        True if created successfully

    Example:
        create_tab("NEWS_SENTIMENT", [
            "Date", "Overall_Score", "Banking_Score",
            "Headline", "Source", "Timestamp"
        ])

        create_tab("FLOORSHEET", [
            "Date", "Symbol", "Buyer_Broker",
            "Seller_Broker", "Quantity", "Price"
        ])
    """
    try:
        if tab_exists(tab_name):
            log.warning(f"Tab already exists: {tab_name}")
            return True

        sheet = _get_sheet()
        worksheet = sheet.add_worksheet(
            title=tab_name,
            rows=1000,
            cols=len(columns) + 5
        )

        # Write header row
        worksheet.append_row(columns, value_input_option="USER_ENTERED")

        # Register in TABS dict dynamically
        key = tab_name.lower().replace(" ", "_")
        TABS[key] = tab_name

        # Register in SCHEMA tab
        if register:
            _register_schema(tab_name, columns, version="1.0")

        log.info(f"Created tab: {tab_name} with {len(columns)} columns")
        return True

    except Exception as e:
        log.error(f"Failed to create tab {tab_name}: {e}")
        return False


def get_schema(tab_name: str) -> dict:
    """
    Get schema info for any tab from SCHEMA registry.

    Args:
        tab_name: Exact tab name

    Returns:
        Dict with columns, version, created date

    Example:
        schema = get_schema("MARKET_LOG")
        print(schema["columns"])
        print(schema["version"])
    """
    try:
        if not tab_exists("SCHEMA"):
            log.warning("SCHEMA tab does not exist yet")
            return {}

        worksheet = _get_tab("SCHEMA")
        records = worksheet.get_all_records()

        for row in records:
            if row.get("Tab_Name", "") == tab_name:
                columns = str(row.get("Columns", "")).split(",")
                return {
                    "tab_name": tab_name,
                    "columns": [c.strip() for c in columns],
                    "version": row.get("Version", "1.0"),
                    "created": row.get("Created", ""),
                    "status": row.get("Status", "ACTIVE"),
                    "last_migrated": row.get("Last_Migrated", "")
                }

        log.warning(f"No schema found for: {tab_name}")
        return {}

    except Exception as e:
        log.error(f"Failed to get schema for {tab_name}: {e}")
        return {}


def get_all_schemas() -> list[dict]:
    """
    Get schema info for all registered tabs.

    Returns:
        List of schema dicts
    """
    try:
        if not tab_exists("SCHEMA"):
            return []
        worksheet = _get_tab("SCHEMA")
        return worksheet.get_all_records()
    except Exception as e:
        log.error(f"Failed to get all schemas: {e}")
        return []


def add_column(tab_name: str, column_name: str,
               default_value: str = "") -> bool:
    """
    Add a single new column to an existing tab.
    Backfills all existing rows with default value.
    Updates SCHEMA registry.

    Args:
        tab_name: Exact tab name
        column_name: New column header
        default_value: Value to fill for existing rows

    Returns:
        True if successful

    Example:
        add_column("MARKET_LOG", "OBV_Trend", "")
        add_column("MARKET_LOG", "Broker_Score", "0")
        add_column("WATCHLIST", "Fundamental_Score", "0")
    """
    try:
        worksheet = _get_tab(tab_name)
        data = worksheet.get_all_values()

        if not data:
            log.error(f"Tab {tab_name} is empty")
            return False

        headers = data[0]

        # Check if column already exists
        if column_name in headers:
            log.warning(f"Column '{column_name}' already exists in {tab_name}")
            return True

        # Find next empty column
        next_col = len(headers) + 1

        # Write header
        worksheet.update_cell(1, next_col, column_name)

        # Backfill existing rows with default
        if len(data) > 1 and default_value != "":
            col_letter = _col_to_letter(next_col)
            start_cell = f"{col_letter}2"
            end_cell = f"{col_letter}{len(data)}"
            fill_values = [[default_value]] * (len(data) - 1)
            worksheet.update(
                f"{start_cell}:{end_cell}",
                fill_values,
                value_input_option="USER_ENTERED"
            )

        # Update SCHEMA
        _update_schema_columns(tab_name, headers + [column_name])

        log.info(f"Added column '{column_name}' to {tab_name}")
        return True

    except Exception as e:
        log.error(f"Failed to add column '{column_name}' to {tab_name}: {e}")
        return False


def migrate_tab(tab_name: str,
                add_columns: list[str] = None,
                default_values: dict = None) -> bool:
    """
    Add multiple new columns to existing tab.
    Full schema migration with version tracking.
    Preserves all existing data.

    Args:
        tab_name: Exact tab name
        add_columns: List of new column names to add
        default_values: Dict of {column: default} for backfill

    Returns:
        True if migration successful

    Example:
        # Add OBV and broker score to MARKET_LOG
        migrate_tab("MARKET_LOG",
            add_columns=["OBV_Trend", "Broker_Score",
                         "Dual_Audit", "GPT_Verdict"],
            default_values={
                "OBV_Trend": "UNKNOWN",
                "Broker_Score": "0",
                "Dual_Audit": "NO",
                "GPT_Verdict": "PENDING"
            }
        )

        # Add fundamental columns to WATCHLIST
        migrate_tab("WATCHLIST",
            add_columns=["Fundamental_Score",
                         "PE_Ratio", "EPS", "ROE"],
            default_values={"Fundamental_Score": "0"}
        )
    """
    if not add_columns:
        log.warning("No columns specified for migration")
        return False

    if default_values is None:
        default_values = {}

    log.info(f"Migrating {tab_name}: adding {add_columns}")
    success_count = 0

    for column in add_columns:
        default = default_values.get(column, "")
        if add_column(tab_name, column, default):
            success_count += 1
        time.sleep(1)  # Avoid API rate limits

    # Bump version in SCHEMA
    _bump_schema_version(tab_name)

    log.info(f"Migration complete: {success_count}/{len(add_columns)} columns added to {tab_name}")
    return success_count == len(add_columns)


def rename_tab(old_name: str, new_name: str) -> bool:
    """
    Rename an existing tab.
    Updates SCHEMA and TABS dict.

    Args:
        old_name: Current tab name
        new_name: New tab name

    Returns:
        True if successful
    """
    try:
        sheet = _get_sheet()
        worksheet = sheet.worksheet(old_name)
        worksheet.update_title(new_name)

        # Update TABS dict
        for key, val in TABS.items():
            if val == old_name:
                TABS[key] = new_name
                break

        # Update SCHEMA
        if tab_exists("SCHEMA"):
            schema_ws = _get_tab("SCHEMA")
            data = schema_ws.get_all_values()
            headers = data[0] if data else []
            if "Tab_Name" in headers:
                col = headers.index("Tab_Name") + 1
                for i, row in enumerate(data[1:], start=2):
                    if row[col - 1] == old_name:
                        schema_ws.update_cell(i, col, new_name)
                        break

        log.info(f"Renamed tab: {old_name} → {new_name}")
        return True

    except Exception as e:
        log.error(f"Failed to rename tab {old_name}: {e}")
        return False


def archive_tab(tab_name: str) -> bool:
    """
    Archive a tab by marking it ARCHIVED in SCHEMA.
    Does not delete data — just flags it inactive.

    Args:
        tab_name: Tab to archive

    Returns:
        True if successful
    """
    try:
        if not tab_exists("SCHEMA"):
            return False

        schema_ws = _get_tab("SCHEMA")
        data = schema_ws.get_all_values()
        headers = data[0] if data else []

        if "Tab_Name" not in headers or "Status" not in headers:
            return False

        tab_col = headers.index("Tab_Name") + 1
        status_col = headers.index("Status") + 1

        for i, row in enumerate(data[1:], start=2):
            if row[tab_col - 1] == tab_name:
                schema_ws.update_cell(i, status_col, "ARCHIVED")
                log.info(f"Archived tab: {tab_name}")
                return True

        return False

    except Exception as e:
        log.error(f"Failed to archive tab {tab_name}: {e}")
        return False


def ensure_tab(tab_name: str, columns: list[str]) -> bool:
    """
    Ensure a tab exists with at minimum the specified columns.
    Creates if missing. Migrates if columns are missing.
    Safe to call every run — idempotent.

    Args:
        tab_name: Exact tab name
        columns: Required columns list

    Returns:
        True if tab ready

    Example:
        # Call at system startup to verify all tabs
        ensure_tab("NEWS_SENTIMENT", [
            "Date", "Score", "Headline"
        ])
        # If tab missing → creates it
        # If tab exists but missing columns → adds them
        # If tab exists with all columns → does nothing
    """
    try:
        if not tab_exists(tab_name):
            log.info(f"Tab missing, creating: {tab_name}")
            return create_tab(tab_name, columns)

        # Tab exists — check for missing columns
        worksheet = _get_tab(tab_name)
        existing_headers = worksheet.row_values(1)
        missing = [c for c in columns if c not in existing_headers]

        if missing:
            log.info(f"Tab {tab_name} missing columns: {missing}")
            return migrate_tab(tab_name, add_columns=missing)

        log.info(f"Tab {tab_name} verified ✓")
        return True

    except Exception as e:
        log.error(f"Failed to ensure tab {tab_name}: {e}")
        return False


def initialize_all_tabs() -> dict:
    """
    Ensure ALL system tabs exist with correct columns.
    Call once at system startup.
    Safe to run multiple times — only creates/migrates what is missing.

    Returns:
        Dict of {tab_name: status}

    Example:
        results = initialize_all_tabs()
        for tab, status in results.items():
            print(f"{tab}: {status}")
    """
    TAB_SCHEMAS = {
        "WATCHLIST": [
            "Symbol", "Company", "Sector", "Added_Date",
            "Fundamental_Score", "Technical_Score",
            "Combined_Score", "Last_Updated",
            "Sector_Momentum", "Dividend_Yield_Pct",
            "PE_Ratio", "EPS", "NPL_Pct", "CAR_Pct",
            "52_Week_High", "52_Week_Low",
            "Pct_From_52W_High", "Notes"
        ],
        "PORTFOLIO": [
            "Symbol", "Entry_Date", "Entry_Price",
            "Shares", "Total_Cost", "Current_Price",
            "Current_Value", "PnL_NPR", "PnL_Pct",
            "Peak_Price", "Stop_Type", "Stop_Level",
            "Trail_Active", "Trail_Stop", "Status",
            "Exit_Date", "Exit_Price", "Exit_Reason"
        ],
        "MARKET_LOG": [
            "Date", "Time", "Symbol", "Sector",
            "Action", "Confidence", "Entry_Price",
            "Stop_Loss", "Target", "Allocation_NPR",
            "Shares", "Breakeven", "Risk_Reward",
            "RSI_14", "EMA_20", "EMA_50", "EMA_200",
            "MACD_Line", "MACD_Signal", "Volume",
            "Volume_Ratio", "OBV_Trend", "VWAP",
            "ATR_14", "Bollinger_Upper", "Bollinger_Lower",
            "Support_Level", "Resistance_Level",
            "Candle_Pattern", "Conf_Score",
            "PE_Ratio", "EPS", "ROE", "NPL_Pct",
            "Fundamental_Score", "Geo_Score", "Macro_Score",
            "Reasoning", "Outcome", "Actual_PnL",
            "Exit_Price", "Exit_Date", "Exit_Reason",
            "Dual_Audit", "GPT_Verdict", "Timestamp"
        ],
        "LEARNING_HUB": [
            "Date", "Symbol", "Sector", "Pattern",
            "Lesson", "Outcome", "PnL_NPR",
            "Confidence", "Applied_Count",
            "Win_When_Applied", "Source", "Timestamp"
        ],
        "MACRO_DATA": [
            "Indicator", "Value", "Unit",
            "Direction", "Impact", "Last_Updated", "Source"
        ],
        "GEOPOLITICAL_DATA": [
            "Date", "Time", "Crude_Price", "Crude_Change_Pct",
            "VIX", "VIX_Level", "Nifty", "Nifty_Change_Pct",
            "DXY", "Gold_Price", "Geo_Score",
            "Status", "Key_Event", "Timestamp"
        ],
        "CANDLE_PATTERNS": [
            "Pattern_Name", "Type", "Tier",
            "Nepal_Win_Rate_Pct", "Sample_Size",
            "Avg_Gain_Pct", "Best_Sector",
            "Best_RSI_Range", "Volume_Condition",
            "Reliability", "Notes"
        ],
        "CAPITAL_ALLOCATION": [
            "Date", "Market_State", "NEPSE_vs_200DMA",
            "Stocks_Pct", "FD_Pct", "Savings_Pct",
            "OD_Pct", "FD_Rate_Used", "Expected_Return",
            "Reasoning", "Review_Date", "Status"
        ],
        "SETTINGS": [
            "Key", "Value", "Description",
            "Last_Updated", "Set_By"
        ],
        "FINANCIALS": [
            "KPI_Name", "Current_Value", "Target_Value",
            "Alert_Level", "Status", "Last_Updated", "Notes"
        ],
        "FUNDAMENTALS": [
            "Symbol", "Company", "Sector", "Quarter",
            "Fiscal_Year", "EPS", "PE_Ratio", "BVPS",
            "PBV_Ratio", "ROE", "ROA", "DPS",
            "Dividend_Yield", "Net_Profit_NPR",
            "Revenue_NPR", "Net_Profit_Growth_Pct",
            "Revenue_Growth_Pct", "Debt_to_Equity",
            "Market_Cap_NPR", "CAR_Pct", "NPL_Pct",
            "NIM_Pct", "CD_Ratio", "Report_Date",
            "Data_Source"
        ],
        "SECTOR_MOMENTUM": [
            "Date", "Sector", "Weekly_Return_Pct",
            "Monthly_Return_Pct", "Avg_Volume_Change_Pct",
            "Momentum_Score", "Status",
            "Top_Stock_1", "Top_Stock_2",
            "Catalyst", "Last_Updated"
        ],
        "NEWS_SENTIMENT": [
            "Date", "Overall_Score", "Banking_Score",
            "Hydro_Score", "Insurance_Score",
            "Microfinance_Score", "Top_Positive_News",
            "Top_Negative_News", "Key_Stock_Mentions",
            "Source_Count", "Timestamp"
        ],
        "CORPORATE_EVENTS": [
            "Symbol", "Company", "Event_Type",
            "Announcement_Date", "Event_Date",
            "Book_Close_Date", "Details",
            "Expected_Impact_Pct", "Days_Until_Event",
            "Status", "Source"
        ],
        "MARKET_BREADTH": [
            "Date", "Advancing", "Declining",
            "Unchanged", "New_52W_High", "New_52W_Low",
            "Total_Turnover_NPR", "Total_Volume",
            "Breadth_Score", "Market_Signal", "Timestamp"
        ],
        "FINANCIAL_ADVISOR": [
            "Date", "Recommendation_Type",
            "Market_Phase", "Confidence_Pct",
            "Capital_In_Stocks_Pct", "Capital_In_FD_Pct",
            "Capital_In_Savings_Pct", "Capital_In_OD_Pct",
            "Three_Month_Outlook", "Expected_Return_Pct",
            "FD_Rate_Used", "Trigger_To_Change",
            "Review_Date", "Actual_Outcome",
            "Was_Forecast_Correct"
        ],
        "BACKTEST_RESULTS": [
            "Test_Name", "Parameter_Tested",
            "Optimal_Value", "Win_Rate_At_Optimal",
            "Sample_Size", "Confidence",
            "Date_Run", "Notes"
        ],
        "SCHEMA": [
            "Tab_Name", "Columns", "Version",
            "Created", "Last_Migrated",
            "Status", "Notes"
        ]
    }

    results = {}
    for tab_name, columns in TAB_SCHEMAS.items():
        try:
            success = ensure_tab(tab_name, columns)
            results[tab_name] = "✓ Ready" if success else "✗ Failed"
            time.sleep(1)  # Avoid API rate limits
        except Exception as e:
            results[tab_name] = f"✗ Error: {e}"

    return results


# ─────────────────────────────────────────
# SCHEMA INTERNAL HELPERS
# ─────────────────────────────────────────
def _register_schema(tab_name: str, columns: list[str],
                     version: str = "1.0") -> bool:
    """Register a tab in the SCHEMA registry."""
    try:
        # Ensure SCHEMA tab exists first
        if not tab_exists("SCHEMA"):
            sheet = _get_sheet()
            ws = sheet.add_worksheet(title="SCHEMA", rows=200, cols=10)
            ws.append_row([
                "Tab_Name", "Columns", "Version",
                "Created", "Last_Migrated", "Status", "Notes"
            ])

        schema_ws = _get_tab("SCHEMA")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check if already registered
        records = schema_ws.get_all_records()
        for row in records:
            if row.get("Tab_Name") == tab_name:
                return True  # Already registered

        # Add new entry
        schema_ws.append_row([
            tab_name,
            ",".join(columns),
            version,
            now,
            now,
            "ACTIVE",
            ""
        ])
        log.info(f"Registered schema for: {tab_name}")
        return True

    except Exception as e:
        log.error(f"Failed to register schema for {tab_name}: {e}")
        return False


def _update_schema_columns(tab_name: str,
                            columns: list[str]) -> bool:
    """Update column list in SCHEMA for a tab."""
    try:
        if not tab_exists("SCHEMA"):
            return False

        schema_ws = _get_tab("SCHEMA")
        data = schema_ws.get_all_values()
        headers = data[0] if data else []

        if "Tab_Name" not in headers:
            return False

        tab_col = headers.index("Tab_Name") + 1
        col_col = headers.index("Columns") + 1 if "Columns" in headers else None
        upd_col = headers.index("Last_Migrated") + 1 if "Last_Migrated" in headers else None
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for i, row in enumerate(data[1:], start=2):
            if row[tab_col - 1] == tab_name:
                if col_col:
                    schema_ws.update_cell(i, col_col, ",".join(columns))
                if upd_col:
                    schema_ws.update_cell(i, upd_col, now)
                return True

        return False

    except Exception as e:
        log.error(f"Failed to update schema columns for {tab_name}: {e}")
        return False


def _bump_schema_version(tab_name: str) -> bool:
    """Increment version number in SCHEMA for a tab."""
    try:
        if not tab_exists("SCHEMA"):
            return False

        schema_ws = _get_tab("SCHEMA")
        data = schema_ws.get_all_values()
        headers = data[0] if data else []

        if "Tab_Name" not in headers or "Version" not in headers:
            return False

        tab_col = headers.index("Tab_Name") + 1
        ver_col = headers.index("Version") + 1
        upd_col = headers.index("Last_Migrated") + 1 if "Last_Migrated" in headers else None
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for i, row in enumerate(data[1:], start=2):
            if row[tab_col - 1] == tab_name:
                try:
                    current = float(row[ver_col - 1])
                    new_version = round(current + 0.1, 1)
                except ValueError:
                    new_version = 1.1
                schema_ws.update_cell(i, ver_col, str(new_version))
                if upd_col:
                    schema_ws.update_cell(i, upd_col, now)
                log.info(f"Version bumped: {tab_name} → {new_version}")
                return True

        return False

    except Exception as e:
        log.error(f"Failed to bump version for {tab_name}: {e}")
        return False


def _col_to_letter(col: int) -> str:
    """Convert column number to letter (1=A, 2=B, 27=AA)."""
    result = ""
    while col > 0:
        col, remainder = divmod(col - 1, 26)
        result = chr(65 + remainder) + result
    return result


# ─────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────
def health_check() -> dict:
    """
    Verify connection to Google Sheets.
    Check all required tabs exist.
    Returns status report.
    """
    result = {
        "connected": False,
        "spreadsheet": None,
        "tabs_found": [],
        "tabs_missing": [],
        "total_tabs": 0
    }

    try:
        sheet = _get_sheet()
        result["connected"] = True
        result["spreadsheet"] = sheet.title

        existing_tabs = [ws.title for ws in sheet.worksheets()]
        result["total_tabs"] = len(existing_tabs)

        for key, tab_name in TABS.items():
            if tab_name in existing_tabs:
                result["tabs_found"].append(tab_name)
            else:
                result["tabs_missing"].append(tab_name)

        log.info(f"Health check: {len(result['tabs_found'])} tabs found, "
                 f"{len(result['tabs_missing'])} missing")
        return result

    except Exception as e:
        log.error(f"Health check failed: {e}")
        result["error"] = str(e)
        return result


# ─────────────────────────────────────────
# TEST / MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "init":
        print("\n" + "="*50)
        print("NEPSE AI — Creating / Verifying Tabs")
        print("="*50)

        results = initialize_all_tabs()
        for tab, status in results.items():
            print(f"{tab}: {status}")

        print("\nDone.\n")

    else:
        print("\n" + "="*50)
        print("NEPSE AI — sheets.py Health Check")
        print("="*50)

        status = health_check()

        print(f"\nConnected:      {status['connected']}")
        print(f"Spreadsheet:    {status.get('spreadsheet', 'N/A')}")
        print(f"Total tabs:     {status['total_tabs']}")
        print(f"Tabs found:     {len(status['tabs_found'])}")
        print(f"Tabs missing:   {len(status['tabs_missing'])}")

        if status["tabs_missing"]:
            print(f"\nMissing tabs:")
            for tab in status["tabs_missing"]:
                print(f"  ✗ {tab}")

        if status["tabs_found"]:
            print(f"\nFound tabs:")
            for tab in status["tabs_found"]:
                print(f"  ✓ {tab}")