var express = require('express');
var app = express();

app.get('/', function (req, res) {
  res.send('Hello World!');
});

app.listen(8888, function () {
 console.log('8888 Port : 서버 실행 중');
});
