import requests  

url = "http://localhost:5000/predict" 
data = {"features": [3, 2, 1500, 1]}  
response = requests.post(url, json=data)  

print(response.json())
