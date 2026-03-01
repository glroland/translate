FILENAME := /opt/app-root/src/INPUT.md

install:
	pip install -r requirements.txt

run_w_granite4:
	cd src && python translate.py --url http://localhost:8000/v1 --key nokeyneeded --model granite-4.0-h-tiny --chunk-size 5000 $(FILENAME)

run_w_gptoss20b:
	cd src && python translate.py --url http://localhost:8000/v1 --key nokeyneeded --model gpt-oss-20b --chunk-size 5000 $(FILENAME)

run_w_llama318b:
	cd src && python translate.py --url http://localhost:8000/v1 --key nokeyneeded --model llama-31-8b  --chunk-size 5000 $(FILENAME)
