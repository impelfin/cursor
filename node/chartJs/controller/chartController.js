const fs = require('fs');
const path = require('path');

exports.getChartData = (req, res) => {
  const modelDir = path.join(__dirname, '../models');
  const files = fs.readdirSync(modelDir).filter(f => f.endsWith('.json'));
  if (files.length === 0) return res.status(404).json({ error: 'No JSON file found' });
  const data = JSON.parse(fs.readFileSync(path.join(modelDir, files[0]), 'utf8'));
  res.json(data);
};
