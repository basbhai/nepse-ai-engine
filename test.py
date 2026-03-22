
import requests




def fetch_fd_rate():
    URL="https://admin.bankbyaj.com/api/v1/category/fd-individual?"

    headers={
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
    "Host": "admin.bankbyaj.com",
    "Origin": "https://bankbyaj.com",
    "Referer": "https://bankbyaj.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    "sec-ch-ua": "\"Chromium\";v=\"146\", \"Not-A.Brand\";v=\"24\", \"Google Chrome\";v=\"146\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\""
  }
    
    session = requests.session()

    response = session.get(URL,headers=headers)
    data=response.json()

    print(data.get('products',[]))



fetch_fd_rate()