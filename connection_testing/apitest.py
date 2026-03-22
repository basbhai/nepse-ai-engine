import requests
import os
from urllib.parse import urljoin

def get_captcha_id():
    """Fetch a new captcha ID from the API."""
    url = "https://tms49.nepsetms.com.np/tmsapi/authApi/captcha/id"
    headers = {
        "Host": "tms49.nepsetms.com.np",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Priority": "u=1, i",
        "Referer": "https://tms49.nepsetms.com.np/login",
        "Sec-Ch-Ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        captcha_id = data.get("id")
        if captcha_id:
            print(f"Captcha ID obtained: {captcha_id}")
            return captcha_id
        else:
            print("No 'id' field in response")
            return None
    except Exception as e:
        print(f"Failed to get captcha ID: {e}")
        return None

def fetch_captcha_image(captcha_id, save_dir="."):
    """Download the captcha image for the given ID and save it."""
    if not captcha_id:
        print("No captcha ID provided.")
        return None

    url = f"https://tms49.nepsetms.com.np/tmsapi/authApi/captcha/image/{captcha_id}"
    headers = {
        "Host": "tms49.nepsetms.com.np",
        "Accept": "image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",  # typical browser Accept for images
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Priority": "u=1, i",
        "Referer": "https://tms49.nepsetms.com.np/login",
        "Sec-Ch-Ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10, stream=True)
        resp.raise_for_status()

        # Determine file extension from Content-Type
        content_type = resp.headers.get("Content-Type", "")
        if "png" in content_type:
            ext = ".png"
        elif "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"
        elif "gif" in content_type:
            ext = ".gif"
        else:
            ext = ".png"  # default

        filename = os.path.join(save_dir, f"captcha_{captcha_id}{ext}")
        with open(filename, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Captcha image saved as: {filename}")
        return filename

    except Exception as e:
        print(f"Failed to download captcha image: {e}")
        return None

if __name__ == "__main__":
    # Step 1: Get a new captcha ID
    captcha_id = get_captcha_id()
    if captcha_id:
        # Step 2: Fetch and save the image
        fetch_captcha_image(captcha_id)