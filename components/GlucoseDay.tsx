// Hero illustration: a day of continuous-glucose readings with the 70–180 mg/dL target range.

const W = 460;
const H = 300;
const X0 = 46;
const X1 = 436;
const Y0 = 62;
const Y1 = 236;
const LO = 50;
const HI = 260;

const x = (h: number) => X0 + (h / 24) * (X1 - X0);
const y = (g: number) => Y1 - ((g - LO) / (HI - LO)) * (Y1 - Y0);

const MEALS = [
  { at: 8, rise: 82, label: "Breakfast" },
  { at: 13.5, rise: 112, label: "Lunch" },
  { at: 20.5, rise: 96, label: "Dinner" },
];

function glucose(h: number) {
  let g = 96 + 14 * Math.exp(-((h - 5.5) ** 2) / 3);
  for (const m of MEALS) {
    const t = h - m.at;
    if (t > 0) g += m.rise * (t / 0.9) * Math.exp(1 - t / 0.9);
  }
  return g + 3 * Math.sin(h * 2.7) + 2 * Math.sin(h * 6.1);
}

const POINTS = Array.from({ length: 97 }, (_, i) => i / 4).map((h) => [x(h), y(glucose(h))] as const);
const PATH = POINTS.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)} ${py.toFixed(1)}`).join(" ");
const NOW = 14.25;

export default function GlucoseDay() {
  const nowG = Math.round(glucose(NOW));
  return (
    <figure className="w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Illustration of a 24-hour glucose trace">
        <defs>
          <clipPath id="above">
            <rect x={X0} y={0} width={X1 - X0} height={y(180)} />
          </clipPath>
        </defs>
        <rect x="0.75" y="0.75" width={W - 1.5} height={H - 1.5} rx="18" fill="#ffffff" stroke="#e3e1db" strokeWidth="1.5" />

        <path d="M28 22c3.4 4.3 6 7.8 6 11.2a6 6 0 0 1-12 0c0-3.4 2.6-6.9 6-11.2Z" fill="#a3333d" />
        <text x="42" y="36" fontSize="10.5" fill="#121417" className="num">Continuous glucose</text>
        <text x={X1} y="36" fontSize="10" fill="#979ea5" textAnchor="end" className="num">mg/dL · 24 h</text>

        <rect x={X0} y={y(180)} width={X1 - X0} height={y(70) - y(180)} fill="#2f7a5b" fillOpacity="0.07" />
        {[70, 180].map((g) => (
          <g key={g}>
            <line x1={X0} x2={X1} y1={y(g)} y2={y(g)} stroke="#2f7a5b" strokeOpacity="0.45" strokeDasharray="3 4" />
            <text x={X0 - 8} y={y(g) + 3} fontSize="9" fill="#2f7a5b" textAnchor="end" className="num">{g}</text>
          </g>
        ))}
        <text x={X0 - 8} y={y(250) + 3} fontSize="9" fill="#979ea5" textAnchor="end" className="num">250</text>
        {[0, 6, 12, 18, 24].map((h) => (
          <g key={h}>
            <line x1={x(h)} x2={x(h)} y1={Y0} y2={Y1} stroke="#e3e1db" strokeOpacity="0.8" />
            <text x={x(h)} y={Y1 + 16} fontSize="9" fill="#979ea5" textAnchor="middle" className="num">
              {String(h).padStart(2, "0")}:00
            </text>
          </g>
        ))}

        <path d={PATH} fill="none" stroke="#121417" strokeWidth="1.8" strokeLinejoin="round" />
        <path d={PATH} fill="none" stroke="#a3333d" strokeWidth="2.2" strokeLinejoin="round" clipPath="url(#above)" />

        {MEALS.map((m) => (
          <g key={m.label}>
            <circle cx={x(m.at)} cy={Y1 + 32} r="3" fill="none" stroke="#5b636b" />
            <text x={x(m.at) + 7} y={Y1 + 35} fontSize="9" fill="#5b636b">{m.label}</text>
          </g>
        ))}

        <line x1={x(NOW)} x2={x(NOW)} y1={y(nowG) + 6} y2={Y1} stroke="#121417" strokeOpacity="0.3" strokeDasharray="2 3" />
        <circle cx={x(NOW)} cy={y(nowG)} r="7" fill="#a3333d" fillOpacity="0.15" />
        <circle cx={x(NOW)} cy={y(nowG)} r="3.5" fill="#a3333d" />
        <g transform={`translate(${x(NOW) + 12} ${y(nowG) - 30})`}>
          <rect width="92" height="24" rx="6" fill="#121417" />
          <text x="10" y="16" fontSize="10.5" fill="#ffffff" className="num">{nowG} · 14:15</text>
        </g>
      </svg>
      <figcaption className="num mt-2.5 flex justify-between text-[10.5px] uppercase tracking-[0.14em] text-faint">
        <span>Target range 70–180 mg/dL</span>
        <span>illustrative</span>
      </figcaption>
    </figure>
  );
}
