const path = require('path');
const fs = require('fs');

exports.getUSData = (req, res) => {
  fs.readFile(path.join(__dirname, '../models/usData.json'), 'utf8', (err, data) => {
    if (err) return res.status(500).json({ error: 'Data load error' });
    res.json(JSON.parse(data));
  });
};

exports.getStateData = (req, res) => {
  fs.readFile(path.join(__dirname, '../models/stateData.json'), 'utf8', (err, data) => {
    if (err) return res.status(500).json({ error: 'Data load error' });
    res.json(JSON.parse(data));
  });
};
