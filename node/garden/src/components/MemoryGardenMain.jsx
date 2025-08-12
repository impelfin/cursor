import React from "react";

const MemoryGardenMain = () => (
  <svg
    width="100%"
    height="100vh"
    viewBox="0 0 800 600"
    style={{
      display: "block",
      background: "linear-gradient(180deg, #f8fafc 0%, #e0e7ef 100%)",
    }}
  >
    {/* 부드러운 구름 */}
    <ellipse cx="200" cy="100" rx="80" ry="30" fill="#f3f6fb" opacity="0.7" />
    <ellipse cx="600" cy="80" rx="100" ry="35" fill="#f3f6fb" opacity="0.6" />
    {/* 둥근 나무 그라데이션 */}
    <defs>
      <radialGradient id="treeGradient" cx="50%" cy="50%" r="70%">
        <stop offset="0%" stopColor="#b7e5c2" />
        <stop offset="100%" stopColor="#6dc7a1" />
      </radialGradient>
      <radialGradient id="flowerGradient" cx="50%" cy="50%" r="80%">
        <stop offset="0%" stopColor="#fff0f6" />
        <stop offset="100%" stopColor="#fbb1c8" />
      </radialGradient>
      <radialGradient id="firefly" cx="50%" cy="50%" r="80%">
        <stop offset="0%" stopColor="#fffde4" stopOpacity="1" />
        <stop offset="100%" stopColor="#fffde4" stopOpacity="0" />
      </radialGradient>
    </defs>
    {/* 나무 줄기 */}
    <rect x="385" y="320" width="30" height="140" rx="15" fill="#a98274" />
    {/* 나무 둥근 잎 */}
    <ellipse
      cx="400"
      cy="260"
      rx="110"
      ry="70"
      fill="url(#treeGradient)"
      filter="url(#shadow1)"
    />
    {/* 꽃들 */}
    <g>
      <circle cx="340" cy="270" r="13" fill="url(#flowerGradient)" />
      <circle cx="460" cy="250" r="10" fill="url(#flowerGradient)" />
      <circle cx="420" cy="310" r="8" fill="url(#flowerGradient)" />
      <circle cx="380" cy="230" r="7" fill="url(#flowerGradient)" />
      <circle cx="430" cy="220" r="6" fill="url(#flowerGradient)" />
    </g>
    {/* 반딧불이 */}
    <g>
      <circle cx="220" cy="180" r="12" fill="url(#firefly)">
        <animate attributeName="cy" values="180;160;180" dur="5s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.7;1;0.7" dur="5s" repeatCount="indefinite" />
      </circle>
      <circle cx="600" cy="200" r="8" fill="url(#firefly)">
        <animate attributeName="cy" values="200;220;200" dur="4s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.5;1;0.5" dur="4s" repeatCount="indefinite" />
      </circle>
      <circle cx="500" cy="120" r="10" fill="url(#firefly)">
        <animate attributeName="cy" values="120;140;120" dur="6s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.6;1;0.6" dur="6s" repeatCount="indefinite" />
      </circle>
    </g>
    {/* 감성적인 텍스트 */}
    <text
      x="50%"
      y="80%"
      textAnchor="middle"
      fontSize="54"
      fill="#7d5a5a"
      fontFamily="'Nanum Pen Script', cursive"
      opacity="0.92"
      style={{ textShadow: "0 2px 8px #fff" }}
    >
      기억의 정원
    </text>
    <text
      x="50%"
      y="87%"
      textAnchor="middle"
      fontSize="22"
      fill="#a98274"
      fontFamily="'Nanum Pen Script', cursive"
      opacity="0.7"
    >
      소중한 추억이 피어나는 곳
    </text>
  </svg>
);

export default MemoryGardenMain;