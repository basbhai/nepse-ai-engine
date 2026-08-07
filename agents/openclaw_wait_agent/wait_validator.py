#!/usr/bin/env python3
"""
NEPSE Wait-Agent: Validates WAIT conditions daily via OpenClaw
"""
import os
import sys
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment
load_dotenv()

REPORTER_API = os.getenv('REPORTER_API_URL', 'http://localhost:8766')
TELEGRAM_BOT = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT = os.getenv('TELEGRAM_CHAT_ID')

def fetch_pending_waits():
    """Fetch all PENDING WAIT records from market_log"""
    query = '''
    SELECT id, symbol, wait_condition_parsed, entry_price, entry_date
    FROM market_log
    WHERE action='WAIT' AND outcome='PENDING'
    ORDER BY entry_date DESC
    '''

    response = requests.post(
        f'{REPORTER_API}/reporter/run',
        json={'sql': query, 'params': {}},
        timeout=10
    )

    if response.status_code != 200:
        print(f'Error fetching PENDING WAITs: {response.text}')
        return []

    return response.json()

def validate_conditions(waits):
    """Check if WAIT conditions have been met"""
    results = []

    for wait in waits:
        symbol = wait['symbol']
        condition = wait.get('wait_condition_parsed', '')

        print(f'\nValidating {symbol}...')
        print(f'  Condition: {condition}')

        # Parse condition and check market data
        # For now, simple check: did price close above entry?
        query = f'''
        SELECT date, close_price
        FROM price_history
        WHERE symbol='{symbol}' AND date > '{wait['entry_date']}'
        ORDER BY date DESC LIMIT 1
        '''

        try:
            response = requests.post(
                f'{REPORTER_API}/reporter/run',
                json={'sql': query, 'params': {}},
                timeout=10
            )

            if response.status_code == 200 and response.json():
                latest = response.json()[0]
                if latest['close_price'] > float(wait['entry_price']):
                    results.append({
                        'id': wait['id'],
                        'symbol': symbol,
                        'status': 'BOUGHT',
                        'exit_price': latest['close_price'],
                        'exit_date': latest['date']
                    })
                    print(f'  ✓ Condition MET: price {latest["close_price"]} > {wait["entry_price"]}')
        except Exception as e:
            print(f'  ✗ Error checking {symbol}: {e}')

    return results

def update_market_log(results):
    """Update market_log with BOUGHT outcome"""
    for result in results:
        query = f'''
        UPDATE market_log
        SET outcome='BOUGHT', exit_price={result['exit_price']}, exit_date='{result['exit_date']}'
        WHERE id={result['id']}
        '''

        try:
            response = requests.post(
                f'{REPORTER_API}/reporter/run',
                json={'sql': query, 'params': {}},
                timeout=10
            )
            print(f'Updated {result["symbol"]}: {result["status"]}')
        except Exception as e:
            print(f'Error updating {result["symbol"]}: {e}')

def send_telegram_alert(results):
    """Send Telegram notification of updates"""
    if not results or not TELEGRAM_BOT or not TELEGRAM_CHAT:
        return

    message = f'🔔 WAIT Validator Report ({datetime.now().strftime("%H:%M NST")})\n\n'
    message += f'✓ {len(results)} conditions met:\n'

    for r in results:
        message += f'  • {r["symbol"]}: {r["exit_price"]} ({r["exit_date"]})\n'

    try:
        requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_BOT}/sendMessage',
            json={'chat_id': TELEGRAM_CHAT, 'text': message},
            timeout=5
        )
    except:
        pass

def main():
    print(f'[{datetime.now().strftime("%H:%M:%S")}] NEPSE Wait-Agent starting...')

    # Fetch pending conditions
    waits = fetch_pending_waits()
    print(f'Found {len(waits)} PENDING WAIT records')

    if not waits:
        print('No pending conditions to validate')
        return

    # Validate each condition
    results = validate_conditions(waits)

    # Update database
    if results:
        print(f'\nUpdating {len(results)} records...')
        update_market_log(results)
        send_telegram_alert(results)
        print('✓ Complete')
    else:
        print('No conditions met yet')

if __name__ == '__main__':
    main()
