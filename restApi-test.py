import requests
import json

market_data_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJsZW1vbi5tYXJrZXRzIiwiaXNzIjoibGVtb24ubWFya2V0cyIsInN1YiI6InVzcl9xeU5jWTExZmZWMzN3U2I0d1QwQ3FaTENKRHJRS3RQVm0wIiwiZXhwIjoxNjc5NDMyOTQ1LCJpYXQiOjE2NjY0NzI5NDUsImp0aSI6ImFwa19xeU5jYk1NSEhrMEpTeUhzWjZHVFd4aHFNQnpZalpxNlZIIiwibW9kZSI6Im1hcmtldF9kYXRhIn0.aYUUSBWzdx1QqdR8dc3ZD6Tm4I9NZmhKCJOs05LG_0k'
request = requests.get("https://data.lemon.markets/v1/instruments",

          headers={"Authorization": f"Bearer {market_data_key}"})

print(request.json())


def listenToMarkets():
    
