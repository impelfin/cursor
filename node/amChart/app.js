const express = require('express');
const path = require('path');
const chartController = require('./controllers/chartController');

const app = express();
const PORT = 8000;

app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/us', chartController.getUSData);
app.get('/api/state', chartController.getStateData);

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
