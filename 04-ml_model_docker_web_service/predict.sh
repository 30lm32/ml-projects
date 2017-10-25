#!/bin/sh

curl -i -H "Content-Type: application/json" \
-H "Accept: application/json" \
-X POST -d \
'{"v1":1,"v2":27.42,"v3":0.00125,"v4":1,"v5":0,"v6":1,"v7":0,"v8":0.25,"v9":0,"v10":0,"v11":0,"v12":1,"v13":0,"v14":720.0,"v15":0}' \
http://localhost:5000/predict