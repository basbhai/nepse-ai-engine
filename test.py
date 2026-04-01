import requests
r=requests.get("https://tms.roadshowsecurities.com.np/atsweb/login")
print(dict(r.cookies))
print(r.headers.get("Set-Cookie"))