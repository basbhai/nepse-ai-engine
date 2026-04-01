import requests
s=requests.Session()
s.get("https://tms.roadshowsecurities.com.np/atsweb/login")
data={"action":"login","format":"json","txtUserName":"20240300188","txtPassword":"C@hange@021"}
headers={"accept":"*/*","content-type":"application/x-www-form-urlencoded","origin":"https://tms.roadshowsecurities.com.np","referer":"https://tms.roadshowsecurities.com.np/atsweb/login"}
r=s.post("https://tms.roadshowsecurities.com.np/atsweb/login",data=data,headers=headers)
print(r.status_code)
print(r.text)
print(dict(r.cookies))