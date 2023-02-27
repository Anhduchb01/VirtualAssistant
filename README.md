# VIRTUAL ASSISTANT
## Download model và lưu vào thư mục models 

https://drive.google.com/drive/folders/1bXWAUECZk4NdOJSXMxKR8UyKQRRJkA0w?usp=share_link

## Tạo môi trường python 
- **Dùng Conda** 
  - Tạo môi trường :
  ```sh
  conda create -n env python==3.10
  ```
  - Activate môi trường :
  ```sh
  conda activate env
  ```
  - Install thư viện :
  ```sh
  pip install -r requirements.txt
  ```
- **Dùng venv** 
  - Tạo môi trường :
  ```sh
  python3 -m venv env
  ```
  - Activate môi trường :
  
  *Window*
  ```sh
  env\Scripts\activate.bat
  ```
  *Linux*
  ```sh
  source env/bin/activate
  ```
  - Install thư viện :
  ```sh
  pip install -r requirements.txt
  ```
## Run app
```sh
cd streamlit
```

```sh
streamlit run main.py
```
