IP="0.0.0.0"
PORT=8001
URL_ROOT="/translator"
CONFIG="./available_models/example.conf.json"

python server.py --ip $IP --port $PORT --url_root $URL_ROOT --config $CONFIG
