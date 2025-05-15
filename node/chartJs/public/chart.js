fetch('/api/data')
  .then(res => res.json())
  .then(data => {
    // data가 배열일 경우 year, amount_per_person 추출
    if (Array.isArray(data)) {
      const labels = data.map(item => item.year);
      const values = data.map(item => item.amount_per_person);
      new Chart(document.getElementById('myChart'), {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: '연도별 1인당 금액',
            data: values,
            fill: true,
            backgroundColor: 'rgba(54, 162, 235, 0.2)', // 아래 영역 색상
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 2
          }]
        },
        options: {
          responsive: true,
          scales: {
            y: { beginAtZero: true }
          }
        }
      });
    } else {
      document.getElementById('myChart').replaceWith('데이터 형식 오류');
      console.error('예상과 다른 데이터 형식', data);
    }
  })
  .catch(err => {
    document.getElementById('myChart').replaceWith('데이터 로드 실패');
    console.error('데이터 로드 실패', err);
  });
