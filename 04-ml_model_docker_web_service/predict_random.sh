#!/bin/sh

curl -i -H "Content-Type: application/json" \
-H "Accept: application/json" \
http://localhost:5000/predict_random
