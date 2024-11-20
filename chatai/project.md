** Project Structure **

- The project structure is organized as follows:

```
chatai
├── frontend
│   ├── public
│   │   └── index.html
│   ├── src
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── index.js
│   │   └── components
│   │       ├── ChatArea.js
│   │       └── Sidebar.js
│   ├── package.json
│   └── .env      # REACT_APP_API_URL=http://localhost:8000
├── backend
│   ├── app
│   │   └── main.py
│   ├── venv
│   ├── requirements.txt
│   └── .env      # OPENAI_API_KEY=sk-proj-1234567890
├── src
|    └── App.js   # main application component
└── project.md
```

** backend installation and usage **

```
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install --upgrade pip
uvicorn app.main:app --reload
CTRL+C to stop
deactivate
```

** frontend installation and usage **

```
cd frontend
npm install
npm start
CTRL+C to stop
```

- open browser and access http://localhost:3000/
