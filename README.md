Create the virtual environment:
python3 -m venv venv


Activate the virtual environment:
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows


Install dependencies inside the virtual environment:
pip install flask flask-socketio torch torchvision sklearn


Create folder “templates” and put home.html and new.html into it.

Run the app inside the virtual environment:
Flask run


