
# ── Stealth Signals ───────────────────────────────────────────────────────────
# Append these endpoints to web_ui/dashboard_api.py
# (paste before the last line of the file)

@app.get("/stealth/watching")
def get_stealth_watching():
    try:
        today = date.today()
        with _db() as cur:
            cur.execute("""
                SELECT * FROM stealth_signals
                WHERE status = 'WATCHING'
                ORDER BY signal_date DESC
            """)
            rows = _rows(cur)

        for r in rows:
            if r.get("signal_date"):
                try:
                    sig_date = date.fromisoformat(str(r["signal_date"]))
                    r["days_watching"] = (today - sig_date).days
                except Exception:
                    r["days_watching"] = None
            else:
                r["days_watching"] = None

        return {"rows": rows, "count": len(rows)}
    except Exception as e:
        log.exception("stealth/watching failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stealth/triggered")
def get_stealth_triggered():
    try:
        today = date.today()
        with _db() as cur:
            cur.execute("""
                SELECT s.*,
                       p.close AS current_price
                FROM stealth_signals s
                LEFT JOIN LATERAL (
                    SELECT COALESCE(NULLIF(close,''), NULLIF(ltp,''))::float AS close
                    FROM price_history
                    WHERE symbol = s.symbol
                      AND close IS NOT NULL AND close != ''
                      AND close ~ '^[0-9]+\\.?[0-9]*$'
                    ORDER BY date DESC
                    LIMIT 1
                ) p ON true
                WHERE s.status = 'TRIGGERED'
                ORDER BY s.trigger_date DESC
            """)
            rows = _rows(cur)

        for r in rows:
            # Days since trigger
            if r.get("trigger_date"):
                try:
                    tdate = date.fromisoformat(str(r["trigger_date"]))
                    r["days_since_trigger"] = (today - tdate).days
                except Exception:
                    r["days_since_trigger"] = None

            # Live return %
            try:
                tp = float(r["trigger_price"]) if r.get("trigger_price") else None
                cp = float(r["current_price"]) if r.get("current_price") else None
                if tp and cp and tp > 0:
                    r["live_return_pct"] = round((cp - tp) / tp * 100, 2)
                else:
                    r["live_return_pct"] = None
            except Exception:
                r["live_return_pct"] = None

        return {"rows": rows, "count": len(rows)}
    except Exception as e:
        log.exception("stealth/triggered failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stealth/history")
def get_stealth_history():
    try:
        with _db() as cur:
            cur.execute("""
                SELECT * FROM stealth_signals
                WHERE status = 'CLOSED'
                ORDER BY close_date DESC
            """)
            rows = _rows(cur)

        for r in rows:
            # Days held (trigger → close)
            try:
                td = date.fromisoformat(str(r["trigger_date"])) if r.get("trigger_date") else None
                cd = date.fromisoformat(str(r["close_date"]))   if r.get("close_date")   else None
                r["days_held"] = (cd - td).days if (td and cd) else None
            except Exception:
                r["days_held"] = None

            # WIN/LOSS
            try:
                rp = float(r["return_pct"]) if r.get("return_pct") else None
                r["result"] = "WIN" if rp and rp > 0 else ("LOSS" if rp and rp < 0 else "BREAKEVEN")
            except Exception:
                r["result"] = None

        # Summary stats
        returns = []
        for r in rows:
            try:
                rp = float(r["return_pct"]) if r.get("return_pct") else None
                if rp is not None:
                    returns.append(rp)
            except Exception:
                pass

        win_rate   = round(sum(1 for x in returns if x > 0) / len(returns) * 100, 1) if returns else None
        avg_return = round(sum(returns) / len(returns), 2) if returns else None
        sorted_ret = sorted(returns)
        median_ret = sorted_ret[len(sorted_ret)//2] if sorted_ret else None

        return {
            "rows":    rows,
            "count":   len(rows),
            "summary": {
                "total_closed":  len(rows),
                "win_rate":      win_rate,
                "avg_return":    avg_return,
                "median_return": median_ret,
            }
        }
    except Exception as e:
        log.exception("stealth/history failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stealth/stats")
def get_stealth_stats():
    try:
        with _db() as cur:
            cur.execute("SELECT * FROM stealth_signals ORDER BY signal_date DESC")
            all_rows = _rows(cur)

        brokers = {}
        total_watching   = 0
        total_triggered  = 0
        total_closed     = 0
        all_returns      = []

        for r in all_rows:
            bid  = str(r.get("broker_id", ""))
            bname = r.get("broker_name", bid)
            status = r.get("status", "")

            if bid not in brokers:
                brokers[bid] = {
                    "broker_id":       bid,
                    "broker_name":     bname,
                    "signal_count":    0,
                    "triggered_count": 0,
                    "closed_count":    0,
                    "returns":         [],
                }

            brokers[bid]["signal_count"] += 1

            if status == "WATCHING":
                total_watching += 1
            elif status == "TRIGGERED":
                total_triggered += 1
                brokers[bid]["triggered_count"] += 1
            elif status == "CLOSED":
                total_closed += 1
                brokers[bid]["triggered_count"] += 1
                brokers[bid]["closed_count"]    += 1
                try:
                    rp = float(r["return_pct"]) if r.get("return_pct") else None
                    if rp is not None:
                        brokers[bid]["returns"].append(rp)
                        all_returns.append(rp)
                except Exception:
                    pass

        # Compute per-broker stats
        broker_stats = []
        for bid, b in brokers.items():
            rets = b.pop("returns", [])
            b["trigger_rate"] = round(b["triggered_count"] / b["signal_count"] * 100, 1) \
                                if b["signal_count"] > 0 else None
            b["avg_return"]   = round(sum(rets) / len(rets), 2) if rets else None
            b["win_rate"]     = round(sum(1 for x in rets if x > 0) / len(rets) * 100, 1) \
                                if rets else None
            # Historical hit rates from grid search
            hit_rates = {"56": 82.4, "48": 80.0, "49": 78.3, "38": 77.8, "58": 60.0}
            b["validated_hit_rate"] = hit_rates.get(bid)
            broker_stats.append(b)

        broker_stats.sort(key=lambda x: -(x.get("validated_hit_rate") or 0))

        # Overall stats
        overall_hit_rate   = round(sum(1 for x in all_returns if x > 0) / len(all_returns) * 100, 1) if all_returns else None
        overall_avg_return = round(sum(all_returns) / len(all_returns), 2) if all_returns else None
        sorted_all         = sorted(all_returns)
        overall_median     = sorted_all[len(sorted_all)//2] if sorted_all else None

        return {
            "broker_stats": broker_stats,
            "overall": {
                "total_signals":  len(all_rows),
                "watching":       total_watching,
                "triggered":      total_triggered,
                "closed":         total_closed,
                "hit_rate":       overall_hit_rate,
                "avg_return":     overall_avg_return,
                "median_return":  overall_median,
            }
        }
    except Exception as e:
        log.exception("stealth/stats failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
