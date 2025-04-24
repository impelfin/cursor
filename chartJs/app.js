const express = require('express');
const path = require('path');
const chartController = require('./controller/chartController');

const app = express();
const PORT = 8000;

app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/data', chartController.getChartData);

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
