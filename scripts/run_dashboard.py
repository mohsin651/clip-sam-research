import _bootstrap
import argparse
from src.dashboard import create_app
if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="outputs/experiment_500")
    p.add_argument("--port",type=int,default=8050)
    args=p.parse_args()
    create_app(args.output).run(host="127.0.0.1",port=args.port,debug=False)
