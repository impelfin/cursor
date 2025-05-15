let usData, stateData, pyramidRoot, mapRoot;

// 피라미드 차트 렌더 함수
function renderPyramid(data, title = '미국 전체') {
  // 비율 계산 (남성, 여성)
  let totalMale = data.reduce((sum, row) => sum + row.male, 0);
  let totalFemale = data.reduce((sum, row) => sum + row.female, 0);
  let percentData = data.map(row => ({
    age: row.age,
    male: -((row.male / totalMale) * 100), 
    female: (row.female / totalFemale) * 100
  }));
  if (pyramidRoot) pyramidRoot.dispose();
  pyramidRoot = am5.Root.new("pyramid");
  pyramidRoot.setThemes([am5themes_Animated.new(pyramidRoot)]);
  let chart = pyramidRoot.container.children.push(am5xy.XYChart.new(pyramidRoot, {
    width: am5.p100,
    paddingLeft: 0,
    paddingRight: 0,
    paddingTop: 0,
    paddingBottom: 0
  }));

  // Y축: 연령대 (좌/우 모두)
  let yAxis = chart.yAxes.push(am5xy.CategoryAxis.new(pyramidRoot, {
    categoryField: "age",
    renderer: am5xy.AxisRendererY.new(pyramidRoot, {
      axisLabelAligned: true,
      opposite: false,
      paddingLeft: 0,
      paddingRight: 0
    })
  }));
  yAxis.data.setAll(percentData);
  let yAxisRight = chart.yAxes.push(am5xy.CategoryAxis.new(pyramidRoot, {
    categoryField: "age",
    renderer: am5xy.AxisRendererY.new(pyramidRoot, {
      axisLabelAligned: true,
      opposite: true,
      paddingLeft: 0,
      paddingRight: 0
    })
  }));
  yAxisRight.data.setAll(percentData);

  // X축: 10% ~ 0% ~ 10% (남성: 왼쪽, 여성: 오른쪽)
  let xAxis = chart.xAxes.push(am5xy.ValueAxis.new(pyramidRoot, {
    min: -10,
    max: 10,
    renderer: am5xy.AxisRendererX.new(pyramidRoot, {
      minGridDistance: 40,
      paddingLeft: 0,
      paddingRight: 0
    })
  }));
  xAxis.set("numberFormatter", am5.NumberFormatter.new(pyramidRoot, {
    numberFormat: "#'%'"
  }));
  // Hide default labels/ticks/grid
  xAxis.get("renderer").labels.template.setAll({ forceHidden: true });
  xAxis.get("renderer").ticks.template.setAll({ forceHidden: true });
  xAxis.get("renderer").grid.template.setAll({ forceHidden: true });

  // Add only custom ticks/labels/grid at exact 2% intervals
  const ticks = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10];
  ticks.forEach(function(val) {
    let range = xAxis.createAxisRange(xAxis.makeDataItem({ value: val }));
    range.get("label").setAll({
      text: Math.abs(val) + "%",
      centerY: am5.p100,
      centerX: am5.p50,
      paddingTop: 10,
      fontSize: 14,
      forceHidden: false
    });
    range.get("grid").setAll({
      strokeOpacity: 0.5,
      forceHidden: false
    });
    range.get("tick").setAll({
      visible: true,
      strokeWidth: 1,
      length: 8,
      forceHidden: false
    });
  });

  // 남성
  let maleSeries = chart.series.push(am5xy.ColumnSeries.new(pyramidRoot, {
    name: "Males",
    yAxis: yAxis,
    xAxis: xAxis,
    valueXField: "male",
    categoryYField: "age",
    fill: am5.color(0x457b9d),
    stroke: am5.color(0x457b9d),
    clustered: false
  }));
  maleSeries.data.setAll(percentData);
  maleSeries.columns.template.setAll({
    tooltipText: ""
  });
  maleSeries.columns.template.adapters.add("tooltipText", function(text, target) {
    if (!target.dataItem) return "";
    const age = target.dataItem.dataContext.age;
    const male = Math.abs(target.dataItem.dataContext.male);
    return `[bold]${age}[/]\n남자: ${male.toFixed(2)}%`;
  });

  let femaleSeries = chart.series.push(am5xy.ColumnSeries.new(pyramidRoot, {
    name: "Females",
    yAxis: yAxisRight,
    xAxis: xAxis,
    valueXField: "female",
    categoryYField: "age",
    fill: am5.color(0xe76f51),
    stroke: am5.color(0xe76f51),
    clustered: false
  }));
  femaleSeries.data.setAll(percentData);
  femaleSeries.columns.template.setAll({
    tooltipText: "[bold]{categoryY}[/]\n여자: {valueX.formatNumber('#.##')}%"
  });

  // 상단 레이블
  chart.children.push(am5.Label.new(pyramidRoot, {
    text: "Males",
    fontSize: 18,
    fill: am5.color(0x457b9d),
    x: am5.percent(0),
    y: -30,
    centerX: am5.percent(0),
    centerY: am5.percent(0)
  }));
  chart.children.push(am5.Label.new(pyramidRoot, {
    text: "Females",
    fontSize: 18,
    fill: am5.color(0xe76f51),
    x: am5.percent(100),
    y: -30,
    centerX: am5.percent(100),
    centerY: am5.percent(0)
  }));

  chart.appear(1000, 100);
}

// 지도 렌더 함수
function renderMap() {
  if (mapRoot) mapRoot.dispose();
  mapRoot = am5.Root.new("usmap");
  mapRoot.setThemes([am5themes_Animated.new(mapRoot)]);
  let chart = mapRoot.container.children.push(am5map.MapChart.new(mapRoot, {
    panX: "none",
    panY: "none",
    projection: am5map.geoAlbersUsa()
  }));
  let usaSeries = chart.series.push(am5map.MapPolygonSeries.new(mapRoot, {
    geoJSON: am5geodata_usaLow
  }));
  usaSeries.mapPolygons.template.setAll({
    tooltipText: "{name}",
    interactive: true
  });

  // 클릭 시 색상 고정, 하나만 선택
  // 모든 주를 파란색으로 초기화
  usaSeries.events.once("datavalidated", function() {
    usaSeries.mapPolygons.each(function(polygon) {
      polygon.set("fill", am5.color(0xa5d8ff));
    });
  });

  let selectedPolygon = null;
  usaSeries.mapPolygons.template.events.on("click", function(ev) {
    // 이전 선택 해제: 파란색으로 복원
    if (selectedPolygon && selectedPolygon !== ev.target) {
      selectedPolygon.set("fill", am5.color(0xa5d8ff));
      selectedPolygon.set("tooltipText", "{name}");
    }
    selectedPolygon = ev.target;
    // 클릭한 주는 남녀 비율에 따라 피라미드 색상으로 강조
    let stateId = ev.target.dataItem.dataContext.id;
    let abbr = stateId.split('-')[1];
    let state = stateData[abbr];
    let highlightColor = am5.color(0x339af0); // fallback
    if (state) {
      let maleSum = state.reduce((sum, row) => sum + row.male, 0);
      let femaleSum = state.reduce((sum, row) => sum + row.female, 0);
      if (maleSum > femaleSum) {
        highlightColor = am5.color(0x457b9d); // 왼쪽 피라미드(남자) 색
      } else if (femaleSum > maleSum) {
        highlightColor = am5.color(0xe76f51); // 오른쪽 피라미드(여자) 색
      }
    }
    ev.target.set("fill", highlightColor);
    // 피라미드 차트도 같이 갱신
    if (stateData[abbr]) renderPyramid(stateData[abbr], abbr);
  });
}

// 데이터 fetch 및 초기화
Promise.all([
  fetch('/api/us').then(r => r.json()),
  fetch('/api/state').then(r => r.json())
]).then(([us, state]) => {
  usData = us;
  stateData = state;
  renderPyramid(usData);
  renderMap();
});
