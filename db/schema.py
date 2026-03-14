"""
db/schema.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Single source of truth for all table definitions.

Every table in the system is defined here as a CREATE TABLE SQL string.
migrations.py reads from TABLE_DDL to apply initial schema.
sheets.py reads from TABLE_COLUMNS to validate inserts dynamically.

Design decisions:
    - All user-data columns are TEXT — flexible, no cast errors on insert.
      Numeric casting happens in indicators.py / filter_engine.py at read time.
    - Every table has: id SERIAL PRIMARY KEY + inserted_at TIMESTAMPTZ.
    - Indexes added on the columns most commonly filtered/sorted.
    - No foreign keys — this is a time-series analytics system, not an OLTP app.
      Speed and flexibility matter more than referential integrity.
─────────────────────────────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────
# TABLE NAME MAPPING
# Keys must match old TABS dict exactly —
# every other module uses these keys.
# ─────────────────────────────────────────
TABS: dict[str, str] = {
    "watchlist":          "watchlist",
    "portfolio":          "portfolio",
    "market_log":         "market_log",
    "learning_hub":       "learning_hub",
    "macro_data":         "macro_data",
    "geo_data":           "geopolitical_data",
    "candle_patterns":    "candle_patterns",
    "capital_allocation": "capital_allocation",
    "settings":           "settings",
    "financials":         "financials",
    "fundamentals":       "fundamentals",
    "sector_momentum":    "sector_momentum",
    "news_sentiment":     "news_sentiment",
    "corporate_events":   "corporate_events",
    "market_breadth":     "market_breadth",
    "financial_advisor":  "financial_advisor",
    "backtest_results":   "backtest_results",
    "indicators":         "indicators",
    "schema":             "db_schema",       # 'schema' is reserved in postgres
}


# ─────────────────────────────────────────
# TABLE DDL
# Used by migrations.py (migration 001)
# ─────────────────────────────────────────
TABLE_DDL: dict[str, str] = {

    "watchlist": """
        CREATE TABLE IF NOT EXISTS watchlist (
            id                  SERIAL PRIMARY KEY,
            symbol              TEXT NOT NULL,
            company             TEXT,
            sector              TEXT,
            added_date          TEXT,
            fundamental_score   TEXT,
            technical_score     TEXT,
            combined_score      TEXT,
            last_updated        TEXT,
            sector_momentum     TEXT,
            dividend_yield_pct  TEXT,
            pe_ratio            TEXT,
            eps                 TEXT,
            npl_pct             TEXT,
            car_pct             TEXT,
            week52_high         TEXT,
            week52_low          TEXT,
            pct_from_52w_high   TEXT,
            notes               TEXT,
            inserted_at         TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_watchlist_symbol
            ON watchlist (symbol);
        CREATE INDEX IF NOT EXISTS ix_watchlist_sector
            ON watchlist (sector);
    """,

    "portfolio": """
        CREATE TABLE IF NOT EXISTS portfolio (
            id              SERIAL PRIMARY KEY,
            symbol          TEXT NOT NULL,
            entry_date      TEXT,
            entry_price     TEXT,
            shares          TEXT,
            total_cost      TEXT,
            current_price   TEXT,
            current_value   TEXT,
            pnl_npr         TEXT,
            pnl_pct         TEXT,
            peak_price      TEXT,
            stop_type       TEXT,
            stop_level      TEXT,
            trail_active    TEXT,
            trail_stop      TEXT,
            status          TEXT DEFAULT 'OPEN',
            exit_date       TEXT,
            exit_price      TEXT,
            exit_reason     TEXT,
            inserted_at     TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS ix_portfolio_symbol
            ON portfolio (symbol);
        CREATE INDEX IF NOT EXISTS ix_portfolio_status
            ON portfolio (status);
    """,

    "market_log": """
        CREATE TABLE IF NOT EXISTS market_log (
            id                  SERIAL PRIMARY KEY,
            date                TEXT,
            time                TEXT,
            symbol              TEXT,
            sector              TEXT,
            action              TEXT,
            confidence          TEXT,
            entry_price         TEXT,
            stop_loss           TEXT,
            target              TEXT,
            allocation_npr      TEXT,
            shares              TEXT,
            breakeven           TEXT,
            risk_reward         TEXT,
            rsi_14              TEXT,
            ema_20              TEXT,
            ema_50              TEXT,
            ema_200             TEXT,
            macd_line           TEXT,
            macd_signal         TEXT,
            volume              TEXT,
            volume_ratio        TEXT,
            obv_trend           TEXT,
            vwap                TEXT,
            atr_14              TEXT,
            bollinger_upper     TEXT,
            bollinger_lower     TEXT,
            support_level       TEXT,
            resistance_level    TEXT,
            candle_pattern      TEXT,
            conf_score          TEXT,
            pe_ratio            TEXT,
            eps                 TEXT,
            roe                 TEXT,
            npl_pct             TEXT,
            fundamental_score   TEXT,
            geo_score           TEXT,
            macro_score         TEXT,
            reasoning           TEXT,
            outcome             TEXT DEFAULT 'PENDING',
            actual_pnl          TEXT,
            exit_price          TEXT,
            exit_date           TEXT,
            exit_reason         TEXT,
            dual_audit          TEXT,
            gpt_verdict         TEXT,
            timestamp           TEXT,
            inserted_at         TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS ix_market_log_symbol
            ON market_log (symbol);
        CREATE INDEX IF NOT EXISTS ix_market_log_date
            ON market_log (date);
        CREATE INDEX IF NOT EXISTS ix_market_log_outcome
            ON market_log (outcome);
    """,

    "learning_hub": """
        CREATE TABLE IF NOT EXISTS learning_hub (
            id               SERIAL PRIMARY KEY,
            date             TEXT,
            symbol           TEXT,
            sector           TEXT,
            pattern          TEXT,
            lesson           TEXT,
            outcome          TEXT,
            pnl_npr          TEXT,
            confidence       TEXT,
            applied_count    TEXT,
            win_when_applied TEXT,
            source           TEXT,
            timestamp        TEXT,
            inserted_at      TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS ix_learning_hub_symbol
            ON learning_hub (symbol);
        CREATE INDEX IF NOT EXISTS ix_learning_hub_pattern
            ON learning_hub (pattern);
    """,

    "macro_data": """
        CREATE TABLE IF NOT EXISTS macro_data (
            id           SERIAL PRIMARY KEY,
            indicator    TEXT NOT NULL,
            value        TEXT,
            unit         TEXT,
            direction    TEXT,
            impact       TEXT,
            last_updated TEXT,
            source       TEXT,
            inserted_at  TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_macro_data_indicator
            ON macro_data (indicator);
    """,

    "geopolitical_data": """
        CREATE TABLE IF NOT EXISTS geopolitical_data (
            id               SERIAL PRIMARY KEY,
            date             TEXT,
            time             TEXT,
            crude_price      TEXT,
            crude_change_pct TEXT,
            vix              TEXT,
            vix_level        TEXT,
            nifty            TEXT,
            nifty_change_pct TEXT,
            dxy              TEXT,
            gold_price       TEXT,
            geo_score        TEXT,
            status           TEXT,
            key_event        TEXT,
            timestamp        TEXT,
            inserted_at      TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS ix_geo_date
            ON geopolitical_data (date);
    """,

    "candle_patterns": """
        CREATE TABLE IF NOT EXISTS candle_patterns (
            id                  SERIAL PRIMARY KEY,
            pattern_name        TEXT NOT NULL,
            type                TEXT,
            tier                TEXT,
            nepal_win_rate_pct  TEXT,
            sample_size         TEXT,
            avg_gain_pct        TEXT,
            best_sector         TEXT,
            best_rsi_range      TEXT,
            volume_condition    TEXT,
            reliability         TEXT,
            notes               TEXT,
            inserted_at         TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_candle_pattern_name
            ON candle_patterns (pattern_name);
    """,

    "capital_allocation": """
        CREATE TABLE IF NOT EXISTS capital_allocation (
            id               SERIAL PRIMARY KEY,
            date             TEXT,
            market_state     TEXT,
            nepse_vs_200dma  TEXT,
            stocks_pct       TEXT,
            fd_pct           TEXT,
            savings_pct      TEXT,
            od_pct           TEXT,
            fd_rate_used     TEXT,
            expected_return  TEXT,
            reasoning        TEXT,
            review_date      TEXT,
            status           TEXT,
            inserted_at      TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS ix_capital_date
            ON capital_allocation (date);
    """,

    "settings": """
        CREATE TABLE IF NOT EXISTS settings (
            id           SERIAL PRIMARY KEY,
            key          TEXT NOT NULL,
            value        TEXT,
            description  TEXT,
            last_updated TEXT,
            set_by       TEXT,
            inserted_at  TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_settings_key
            ON settings (key);
    """,

    "financials": """
        CREATE TABLE IF NOT EXISTS financials (
            id           SERIAL PRIMARY KEY,
            kpi_name     TEXT NOT NULL,
            current_value TEXT,
            target_value  TEXT,
            alert_level   TEXT,
            status        TEXT,
            last_updated  TEXT,
            notes         TEXT,
            inserted_at   TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_financials_kpi
            ON financials (kpi_name);
    """,

    "fundamentals": """
        CREATE TABLE IF NOT EXISTS fundamentals (
            id                    SERIAL PRIMARY KEY,
            symbol                TEXT NOT NULL,
            company               TEXT,
            sector                TEXT,
            quarter               TEXT,
            fiscal_year           TEXT,
            eps                   TEXT,
            pe_ratio              TEXT,
            bvps                  TEXT,
            pbv_ratio             TEXT,
            roe                   TEXT,
            roa                   TEXT,
            dps                   TEXT,
            dividend_yield        TEXT,
            net_profit_npr        TEXT,
            revenue_npr           TEXT,
            net_profit_growth_pct TEXT,
            revenue_growth_pct    TEXT,
            debt_to_equity        TEXT,
            market_cap_npr        TEXT,
            car_pct               TEXT,
            npl_pct               TEXT,
            nim_pct               TEXT,
            cd_ratio              TEXT,
            report_date           TEXT,
            data_source           TEXT,
            inserted_at           TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS ix_fundamentals_symbol
            ON fundamentals (symbol);
        CREATE INDEX IF NOT EXISTS ix_fundamentals_quarter
            ON fundamentals (quarter, fiscal_year);
    """,

    "sector_momentum": """
        CREATE TABLE IF NOT EXISTS sector_momentum (
            id                   SERIAL PRIMARY KEY,
            date                 TEXT,
            sector               TEXT,
            weekly_return_pct    TEXT,
            monthly_return_pct   TEXT,
            avg_volume_change_pct TEXT,
            momentum_score       TEXT,
            status               TEXT,
            top_stock_1          TEXT,
            top_stock_2          TEXT,
            catalyst             TEXT,
            last_updated         TEXT,
            inserted_at          TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS ix_sector_momentum_date
            ON sector_momentum (date);
        CREATE INDEX IF NOT EXISTS ix_sector_momentum_sector
            ON sector_momentum (sector);
    """,

    "news_sentiment": """
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id                  SERIAL PRIMARY KEY,
            date                TEXT,
            overall_score       TEXT,
            banking_score       TEXT,
            hydro_score         TEXT,
            insurance_score     TEXT,
            microfinance_score  TEXT,
            top_positive_news   TEXT,
            top_negative_news   TEXT,
            key_stock_mentions  TEXT,
            source_count        TEXT,
            timestamp           TEXT,
            inserted_at         TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS ix_news_date
            ON news_sentiment (date);
    """,

    "corporate_events": """
        CREATE TABLE IF NOT EXISTS corporate_events (
            id                   SERIAL PRIMARY KEY,
            symbol               TEXT,
            company              TEXT,
            event_type           TEXT,
            announcement_date    TEXT,
            event_date           TEXT,
            book_close_date      TEXT,
            details              TEXT,
            expected_impact_pct  TEXT,
            days_until_event     TEXT,
            status               TEXT,
            source               TEXT,
            inserted_at          TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS ix_corp_events_symbol
            ON corporate_events (symbol);
        CREATE INDEX IF NOT EXISTS ix_corp_events_date
            ON corporate_events (event_date);
    """,

    "market_breadth": """
        CREATE TABLE IF NOT EXISTS market_breadth (
            id                SERIAL PRIMARY KEY,
            date              TEXT,
            advancing         TEXT,
            declining         TEXT,
            unchanged         TEXT,
            new_52w_high      TEXT,
            new_52w_low       TEXT,
            total_turnover_npr TEXT,
            total_volume      TEXT,
            breadth_score     TEXT,
            market_signal     TEXT,
            nepse_index       TEXT,
            nepse_change_pct  TEXT,
            timestamp         TEXT,
            inserted_at       TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_market_breadth_date
            ON market_breadth (date);
    """,

    "financial_advisor": """
        CREATE TABLE IF NOT EXISTS financial_advisor (
            id                      SERIAL PRIMARY KEY,
            date                    TEXT,
            recommendation_type     TEXT,
            market_phase            TEXT,
            confidence_pct          TEXT,
            capital_in_stocks_pct   TEXT,
            capital_in_fd_pct       TEXT,
            capital_in_savings_pct  TEXT,
            capital_in_od_pct       TEXT,
            three_month_outlook     TEXT,
            expected_return_pct     TEXT,
            fd_rate_used            TEXT,
            trigger_to_change       TEXT,
            review_date             TEXT,
            actual_outcome          TEXT,
            was_forecast_correct    TEXT,
            inserted_at             TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS ix_fin_advisor_date
            ON financial_advisor (date);
    """,

    "backtest_results": """
        CREATE TABLE IF NOT EXISTS backtest_results (
            id                  SERIAL PRIMARY KEY,
            test_name           TEXT,
            parameter_tested    TEXT,
            optimal_value       TEXT,
            win_rate_at_optimal TEXT,
            sample_size         TEXT,
            confidence          TEXT,
            date_run            TEXT,
            notes               TEXT,
            inserted_at         TIMESTAMPTZ DEFAULT NOW()
        );
    """,

    "indicators": """
        CREATE TABLE IF NOT EXISTS indicators (
            id               SERIAL PRIMARY KEY,
            symbol           TEXT NOT NULL,
            date             TEXT NOT NULL,
            ltp              TEXT,
            prev_close       TEXT,
            volume           TEXT,
            history_days     TEXT,
            rsi_14           TEXT,
            rsi_signal       TEXT,
            ema_20           TEXT,
            ema_50           TEXT,
            ema_200          TEXT,
            ema_trend        TEXT,
            ema_20_50_cross  TEXT,
            ema_50_200_cross TEXT,
            macd_line        TEXT,
            macd_signal      TEXT,
            macd_histogram   TEXT,
            macd_cross       TEXT,
            bb_upper         TEXT,
            bb_middle        TEXT,
            bb_lower         TEXT,
            bb_width         TEXT,
            bb_pct_b         TEXT,
            bb_signal        TEXT,
            atr_14           TEXT,
            atr_pct          TEXT,
            obv              TEXT,
            obv_trend        TEXT,
            tech_score       TEXT,
            tech_signal      TEXT,
            timestamp        TEXT,
            inserted_at      TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_indicators_symbol_date
            ON indicators (symbol, date);
        CREATE INDEX IF NOT EXISTS ix_indicators_date
            ON indicators (date);
        CREATE INDEX IF NOT EXISTS ix_indicators_tech_signal
            ON indicators (tech_signal);
    """,

    "db_schema": """
        CREATE TABLE IF NOT EXISTS db_schema (
            id             SERIAL PRIMARY KEY,
            migration_id   TEXT NOT NULL,
            name           TEXT NOT NULL,
            applied_at     TIMESTAMPTZ DEFAULT NOW(),
            rolled_back_at TIMESTAMPTZ,
            status         TEXT DEFAULT 'applied',
            notes          TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_schema_migration_id
            ON db_schema (migration_id);
    """,
}


# ─────────────────────────────────────────
# COLUMN LISTS
# Used by write_row() for dynamic inserts.
# Keys match TABS values (table names).
# Order matches DDL above.
# ─────────────────────────────────────────
TABLE_COLUMNS: dict[str, list[str]] = {
    "watchlist": [
        "symbol", "company", "sector", "added_date",
        "fundamental_score", "technical_score", "combined_score",
        "last_updated", "sector_momentum", "dividend_yield_pct",
        "pe_ratio", "eps", "npl_pct", "car_pct",
        "week52_high", "week52_low", "pct_from_52w_high", "notes",
    ],
    "portfolio": [
        "symbol", "entry_date", "entry_price", "shares", "total_cost",
        "current_price", "current_value", "pnl_npr", "pnl_pct",
        "peak_price", "stop_type", "stop_level", "trail_active",
        "trail_stop", "status", "exit_date", "exit_price", "exit_reason",
    ],
    "market_log": [
        "date", "time", "symbol", "sector", "action", "confidence",
        "entry_price", "stop_loss", "target", "allocation_npr",
        "shares", "breakeven", "risk_reward", "rsi_14", "ema_20",
        "ema_50", "ema_200", "macd_line", "macd_signal", "volume",
        "volume_ratio", "obv_trend", "vwap", "atr_14",
        "bollinger_upper", "bollinger_lower", "support_level",
        "resistance_level", "candle_pattern", "conf_score",
        "pe_ratio", "eps", "roe", "npl_pct", "fundamental_score",
        "geo_score", "macro_score", "reasoning", "outcome",
        "actual_pnl", "exit_price", "exit_date", "exit_reason",
        "dual_audit", "gpt_verdict", "timestamp",
    ],
    "learning_hub": [
        "date", "symbol", "sector", "pattern", "lesson",
        "outcome", "pnl_npr", "confidence", "applied_count",
        "win_when_applied", "source", "timestamp",
    ],
    "macro_data": [
        "indicator", "value", "unit", "direction",
        "impact", "last_updated", "source",
    ],
    "geopolitical_data": [
        "date", "time", "crude_price", "crude_change_pct",
        "vix", "vix_level", "nifty", "nifty_change_pct",
        "dxy", "gold_price", "geo_score", "status",
        "key_event", "timestamp",
    ],
    "candle_patterns": [
        "pattern_name", "type", "tier", "nepal_win_rate_pct",
        "sample_size", "avg_gain_pct", "best_sector",
        "best_rsi_range", "volume_condition", "reliability", "notes",
    ],
    "capital_allocation": [
        "date", "market_state", "nepse_vs_200dma", "stocks_pct",
        "fd_pct", "savings_pct", "od_pct", "fd_rate_used",
        "expected_return", "reasoning", "review_date", "status",
    ],
    "settings": [
        "key", "value", "description", "last_updated", "set_by",
    ],
    "financials": [
        "kpi_name", "current_value", "target_value",
        "alert_level", "status", "last_updated", "notes",
    ],
    "fundamentals": [
        "symbol", "company", "sector", "quarter", "fiscal_year",
        "eps", "pe_ratio", "bvps", "pbv_ratio", "roe", "roa",
        "dps", "dividend_yield", "net_profit_npr", "revenue_npr",
        "net_profit_growth_pct", "revenue_growth_pct", "debt_to_equity",
        "market_cap_npr", "car_pct", "npl_pct", "nim_pct",
        "cd_ratio", "report_date", "data_source",
    ],
    "sector_momentum": [
        "date", "sector", "weekly_return_pct", "monthly_return_pct",
        "avg_volume_change_pct", "momentum_score", "status",
        "top_stock_1", "top_stock_2", "catalyst", "last_updated",
    ],
    "news_sentiment": [
        "date", "overall_score", "banking_score", "hydro_score",
        "insurance_score", "microfinance_score", "top_positive_news",
        "top_negative_news", "key_stock_mentions", "source_count", "timestamp",
    ],
    "corporate_events": [
        "symbol", "company", "event_type", "announcement_date",
        "event_date", "book_close_date", "details",
        "expected_impact_pct", "days_until_event", "status", "source",
    ],
    "market_breadth": [
        "date", "advancing", "declining", "unchanged",
        "new_52w_high", "new_52w_low", "total_turnover_npr",
        "total_volume", "breadth_score", "market_signal",
        "nepse_index", "nepse_change_pct", "timestamp",
    ],
    "financial_advisor": [
        "date", "recommendation_type", "market_phase", "confidence_pct",
        "capital_in_stocks_pct", "capital_in_fd_pct",
        "capital_in_savings_pct", "capital_in_od_pct",
        "three_month_outlook", "expected_return_pct", "fd_rate_used",
        "trigger_to_change", "review_date", "actual_outcome",
        "was_forecast_correct",
    ],
    "backtest_results": [
        "test_name", "parameter_tested", "optimal_value",
        "win_rate_at_optimal", "sample_size", "confidence",
        "date_run", "notes",
    ],
    "indicators": [
        "symbol", "date", "ltp", "prev_close", "volume", "history_days",
        "rsi_14", "rsi_signal", "ema_20", "ema_50", "ema_200",
        "ema_trend", "ema_20_50_cross", "ema_50_200_cross",
        "macd_line", "macd_signal", "macd_histogram", "macd_cross",
        "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "bb_pct_b", "bb_signal", "atr_14", "atr_pct",
        "obv", "obv_trend", "tech_score", "tech_signal", "timestamp",
    ],
    "db_schema": [
        "migration_id", "name", "applied_at", "rolled_back_at",
        "status", "notes",
    ],
}