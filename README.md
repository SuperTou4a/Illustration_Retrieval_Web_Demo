# A web demo for automatic illustration retrieval
# Introduction
The demo consists of four parts: NER, web crawler, object detection and style-transfer. Given an English paragraph and a subject name, it could provide you serveral illustrations with consistent style.

# Tip
1.The web demo is currently deployed on blackhole server. You could visit it through 202.120.38.146:9751 with a SJTU VPN.

2.If you want to run it on your own server, you need to modify the port setting in server.py & 1.html and the proxy setting in google_images_download.py. Run server.py to launch the web demo.

3.You aldo need to download the file "yolov5x.pt" by click [this](https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5x.pt) and put it in the root folder. Otherwise server.py would not work.
